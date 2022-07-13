import os
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test
from daisy.utils.dataset import BasicDataset, UAEData
from daisy.utils.config import init_seed, model_config
from daisy.utils.loader import Interactions, get_ur, convert_npy_mat, build_candidates_set, get_adj_mat
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k


if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = dict()
    basic_conf = yaml.load(open('./daisy/config/basic.yaml'), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    args = parse_args()
    algo_name = config['algo_name'] if args.algo_name is None else args.algo_name

    model_conf = yaml.load(
        open(f'./daisy/config/model/{algo_name}.yaml'), Loader=yaml.loader.SafeLoader)
    config.update(model_conf)

    args_conf = vars(args)
    config.update(args_conf)

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])
    
    ''' Test Process for Metrics Exporting '''
    inter = Interactions(config)
    df = inter.get_data()
    user_num, item_num = inter.user_num, inter.item_num
    config['user_num'] = user_num
    config['item_num'] = item_num

    ''' Train Test split '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)

    ''' get ground truth '''
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    config['train_ur'] = total_train_ur

    ''' select model '''
    model = model_config[config['algo_name']](config)

    ''' build and train model '''
    s_time = time.time()
    if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop']:
        model.fit(train_set)
    elif config['algo_name'].lower() in ['multi-vae']:
        # TODO
        train_dataset = UAEData(user_num, item_num, train_set, test_set)
        training_mat = convert_npy_mat(user_num, item_num, train_set)
    elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf']:
        if config['algo_name'].lower() == 'ngcf':
            _, norm_adj, _ = get_adj_mat(user_num,item_num)
            config['norm_adj'] = norm_adj

        sampler = BasicNegtiveSampler(train_set, total_train_ur, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)
    elif config['algo_name'].lower() in ['item2vec']:
        sampler = SkipGramNegativeSampler(train_set, total_train_ur, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)
    else:
        raise NotImplementedError('Something went wrong when building and training...')
    elapsed_time = time.time() - s_time
    print(f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

    ''' build candidates set '''
    print('Start Calculating Metrics...')
    test_ucands = build_candidates_set(test_ur, total_train_ur, config)
    



    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    if args.algo_name in ['multi-vae', 'cdae', 'itemknn', 'puresvd', 'slim'] and args.problem_type == 'point-wise':
        for u in tqdm(test_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in test_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_ucands[u])[rec_idx]
            preds[u] = top_n
    elif args.algo_name in ['mostpop']:
        preds = model.predict(test_ur, total_train_ur, args.topk)
    else:
        for u in tqdm(test_ucands.keys()):
            # build a test MF dataset for certain user u to accelerate
            tmp = pd.DataFrame({
                'user': [u for _ in test_ucands[u]], 
                'item': test_ucands[u], 
                'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
            })
            tmp_neg_set = sampler.transform(tmp, is_training=False)
            tmp_dataset = PairData(tmp_neg_set, is_training=False)
            tmp_loader = DataLoader(
                tmp_dataset,
                batch_size=candidates_num, 
                shuffle=False, 
                num_workers=0
            )
            # get top-N list with torch method 
            for items in tmp_loader:
                user_u, item_i = items[0], items[1]
                if torch.cuda.is_available():
                    user_u = user_u.cuda()
                    item_i = item_i.cuda()
                else:
                    user_u = user_u.cpu()
                    item_i = item_i.cpu()

                prediction = model.predict(user_u, item_i)
                _, indices = torch.topk(prediction, args.topk)
                top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

            preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/{args.prepro}/{args.test_method}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            print(f'Precision@{k}: {pre_k:.4f}')
            print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            print(f'MAP@{k}: {map_k:.4f}')
            print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    common_prefix = f'with_{args.sample_ratio}{args.sample_method}'
    algo_prefix = f'{args.loss_type}_{args.problem_type}_{args.algo_name}'

    res.to_csv(
        f'{result_save_path}{algo_prefix}_{common_prefix}_kpi_results.csv', 
        index=False
    )
