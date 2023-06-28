import json
import optuna
import numpy as np
from logging import getLogger
from daisy.model.Models import RecommenderModel
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.config import init_seed, init_config, init_logger, param_type_config
from daisy.utils.metrics import metrics_config
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix


TRIAL_CNT = 0

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

    ''' unpack hyperparameters to tune '''
    param_dict = json.loads(config['tune_pack'])
    kpi_name = config['optimization_metric']
    algo_name = config['algo_name'].lower()
    tune_param_names = RecommenderModel(algo_name).tunable_param_names

    ''' open logfile to record tuning process '''
    # begin tuning here
    tune_log_path = './tune_res/'
    ensure_dir(tune_log_path)

    f = open(tune_log_path + f"best_params_{config['loss_type']}_{config['algo_name']}_{config['dataset']}_{config['prepro']}_{config['val_method']}.csv", 'w', encoding='utf-8')
    line = ','.join(tune_param_names) + f',{kpi_name}'
    f.write(line + '\n')
    f.flush()

    ''' Test Process for Metrics Exporting '''
    reader, processor = RawDataReader(config), Preprocessor(config)
    df = reader.get_data()
    df = processor.process(df)
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num # Total number of users
    config['item_num'] = item_num # Total number of items

    ''' Train Test split '''
    splitter = TestSplitter(config)
    train_index, test_index = splitter.split(df)
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

    ''' define optimization target function '''
    def objective(trial):
        global TRIAL_CNT
        for param in tune_param_names:
            if param not in param_dict.keys(): continue
                
            if isinstance(param_dict[param], list):
                config[param] = trial.suggest_categorical(param, param_dict[param])
            elif isinstance(param_dict[param], dict):
                if param_type_config[param] == 'int':
                    step = param_dict[param]['step']
                    config[param] = trial.suggest_int(
                        param, param_dict[param]['min'], param_dict[param]['max'], 1 if step is None else step)
                elif param_type_config[param] == 'float':
                    config[param] = trial.suggest_float(
                        param, param_dict[param]['min'], param_dict[param]['max'], step=param_dict[param]['step'])
                else:
                    raise ValueError(f'Invalid parameter type for {param}...')
            else:
                raise ValueError(f'Invalid parameter settings for {param}, Current is {param_dict[param]}...')
        
        ''' user train set to get validation combinations and build model for each dataset '''
        splitter = ValidationSplitter(config)
        cnt, kpis = 1, []
        for train_index, val_index in splitter.split(train_set):
            train, validation = train_set.iloc[train_index, :].copy(), train_set.iloc[val_index, :].copy()

            ''' get ground truth '''
            val_ur = get_ur(validation)
            train_ur = get_ur(train)
            config['train_ur'] = train_ur

            ''' build and train model '''
            model = RecommenderModel(config['algo_name'])(config)
            if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop', 'sndmostpop','ease']:
                model.fit(train)
            
            elif config['algo_name'].lower() in ['multi-vae']:
                history_item_id, history_item_value, _  = get_history_matrix(train, config, row='user')
                config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
                train_dataset = AEDataset(train, yield_col=config['UID_NAME'])
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)

            elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
                if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
                    config['inter_matrix'] = get_inter_matrix(train, config)
                sampler = BasicNegtiveSampler(train, config)
                train_samples = sampler.sampling()
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)

            elif config['algo_name'].lower() in ['item2vec']:
                sampler = SkipGramNegativeSampler(train, config)
                train_samples = sampler.sampling()
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)
            else:
                raise NotImplementedError('Something went wrong when building and training...')
            logger.info(f'Finish {cnt} train-validation experiment(s)...')
            cnt += 1

            ''' build candidates set '''
            logger.info('Start Calculating Metrics...')
            val_u, val_ucands = build_candidates_set(val_ur, train_ur, config)

            ''' get predict result '''
            logger.info('==========================')
            logger.info('Generate recommend list...')
            logger.info('==========================')
            val_dataset = CandidatesDataset(val_ucands)
            val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
            preds = model.rank(val_loader) 

            ''' calculating KPIs '''
            kpi = metrics_config[kpi_name]['evaluator'](val_ur, preds, val_u)
            kpis.append(kpi)
        
        TRIAL_CNT += 1
        logger.info(f'Finish {TRIAL_CNT} trial...')

        return np.mean(kpis)

    ''' init optuna workspace '''
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2022))
    study.optimize(objective, n_trials=config['hyperopt_trail'])

    ''' record the best choices '''
    logger.info(f'Trial {study.best_trial.number} get the best {kpi_name}({study.best_trial.value}) with params: {study.best_trial.params}')
    line = ','.join([str(study.best_params[param]) if param in param_dict.keys() else str(config[param]) for param in tune_param_names]) + f',{study.best_value:.4f}\n'
    f.write(line)
    f.flush()
    f.close()
