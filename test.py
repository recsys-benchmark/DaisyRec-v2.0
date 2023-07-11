import time
from logging import getLogger
from daisy.model.Models import RecommenderModel
from daisy.utils.splitter import TestSplitter
from daisy.utils.metrics import calc_ranking_results
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix, get_ur_new
from sys import exit

def sampler_compare(train_set, config):

    for num_neg in 4, 5, 6:
        print("\n======================================")
        print(f'Now calculating for {num_neg} negative samples per user-item pair:')

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.sampling()
        e = time.time()
        old_time = e-s
        print(f'Prev sampling time: {round(old_time, 5)}')
        del sampler
        
        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.guess_and_check_sampling()
        e = time.time()
        guessandcheck_time = e-s
        print(f'Guess and check sampling time: {round(guessandcheck_time,3)}s')
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.set_diff_sampling()
        e = time.time()
        setdifftime = e-s
        print(f'Set diff sampling time: {round(setdifftime, 4)}')
        del sampler


        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=64)
        e = time.time()
        new_64_time = e-s
        print(f'Batch-aware sampling time: {round(new_64_time,3)}s (batch-size=64)')
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=128)
        e = time.time()
        new_128_time = e-s
        print(f'Batch-aware sampling time: {round(new_128_time,3)}s (batch-size=128)')
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=256)
        e = time.time()
        new_256_time = e-s
        print(f'Batch-aware sampling time: {round(new_256_time,3)}s  (batch-size=256)')
        del sampler

        print(f'\nFor {num_neg} negative items per user-item pair: ')
        print(f'The guess-and-check sampling time is {round(guessandcheck_time/old_time, 2)} times slower')
        print(f'The set difference  sampling time is {round(setdifftime/old_time, 2)} times slower')
        print(f'The  64 batch-aware sampling time is {round(new_64_time/old_time, 2)} times slower')
        print(f'The 128 batch-aware sampling time is {round(new_128_time/old_time, 2)} times slower')
        print(f'The 256 batch-aware sampling time is {round(new_256_time/old_time, 2)} times slower')

    exit()


if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config()

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger

    ''' Test Process for Metrics Exporting '''
    reader, processor = RawDataReader(config), Preprocessor(config)
    df = reader.get_data()
    df = processor.process(df)
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num
    config['item_num'] = item_num

    ''' Train Test split '''
    splitter = TestSplitter(config)
    train_index, test_index = splitter.split(df)
    train_set, test_set = df.iloc[train_index,
                                  :].copy(), df.iloc[test_index, :].copy()

    ''' get ground truth '''
    test_ur = get_ur(test_set)

    s = time.time()
    total_train_ur = get_ur(train_set)
    e = time.time()
    print(f"old time: {e - s}")

    s = time.time()
    total_train_ur = get_ur_new(train_set, item_num)
    e = time.time()
    print(f"new time: {e - s}")

    config['train_ur'] = total_train_ur

    # TODO: REMOVE, FOR DEBUGGING ONLY
    sampler_compare(train_set, config)

    ''' build and train model '''
    s_time = time.time()
    if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
        model = RecommenderModel(config['algo_name'])(config)
        model.fit(train_set)

    elif config['algo_name'].lower() in ['multi-vae']:
        history_item_id, history_item_value, _ = get_history_matrix(
            train_set, config, row='user')
        config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
        model = RecommenderModel(config['algo_name'])(config)
        train_dataset = AEDataset(train_set, yield_col=config['UID_NAME'])
        train_loader = get_dataloader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
        if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
            config['inter_matrix'] = get_inter_matrix(train_set, config)
        model = RecommenderModel(config['algo_name'])(config)
        sampler = BasicNegtiveSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = get_dataloader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    elif config['algo_name'].lower() in ['item2vec']:
        model = RecommenderModel(config['algo_name'])(config)
        sampler = SkipGramNegativeSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = get_dataloader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    else:
        raise NotImplementedError(
            'Something went wrong when building and training...')
    elapsed_time = time.time() - s_time
    logger.info(
        f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

    ''' build candidates set '''
    logger.info('Start Calculating Metrics...')
    test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)

    ''' get predict result '''
    logger.info('==========================')
    logger.info('Generate recommend list...')
    logger.info('==========================')
    test_dataset = CandidatesDataset(test_ucands)
    test_loader = get_dataloader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0)
    preds = model.rank(test_loader)  # np.array (u, topk)

    ''' calculating KPIs '''
    logger.info('Save metric@k result to res folder...')
    result_save_path = f"./res/{config['dataset']}/{config['prepro']}/{config['test_method']}/"
    algo_prefix = f"{config['loss_type']}_{config['algo_name']}"
    common_prefix = f"with_{config['sample_ratio']}{config['sample_method']}"

    ensure_dir(result_save_path)
    config['res_path'] = result_save_path

    results = calc_ranking_results(test_ur, preds, test_u, config)
    results.to_csv(
        f'{result_save_path}{algo_prefix}_{common_prefix}_kpi_results.csv', index=False)
