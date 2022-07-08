import torch
import torch.nn as nn
import torch.optim as optim

algo_config = {
    'itemknn': ['maxk'],
    'puresvd': ['factors'],
    'slim': ['alpha', 'elastic'],
    'mf': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'fm': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'neumf': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'nfm': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'ngcf': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'],
    'multi-vae': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_2', 'kl_reg', 'reg_1']
}

model_config = {
    'initializer':{'normal':{'mean':0.0, 'std':0.01},
                   'uniform':{'a':0.0, 'b':1.0},
                   'xavier_normal':{'gain':1.0},
                   'xavier_uniform':{'gain':1.0}
                   }
}

initializer_config = {'normal': nn.init.normal_,
                     'uniform': nn.init.uniform_,
                     'xavier_normal': nn.init.xavier_normal_,
                     'xavier_uniform': nn.init.xavier_uniform_
                     }

optimizer_config = {'sgd': optim.SGD,
                    'adam': optim.Adam,
                    'adagrad':optim.Adagrad
                   }


import yaml
import random
import numpy as np

config = yaml.load(open(), Loader=yaml.loader.SafeLoader)

def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False