import random
import numpy as np

import torch
import torch.nn as nn

from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.model.PureSVDRecommender import PureSVD
from daisy.model.SLiMRecommender import SLiM
from daisy.model.PopRecommender import MostPop
from daisy.model.MFRecommender import MF
from daisy.model.FMRecommender import FM
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.model.NeuMFRecommender import NeuMF
from daisy.model.NFMRecommender import NFM
from daisy.model.NGCFRecommender import NGCF
from daisy.model.VAERecommender import VAE

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
    'mostpop': MostPop,
    'slim': SLiM,
    'itemknn': ItemKNNCF,
    'puresvd': PureSVD,
    'mf': MF,
    'fm': FM,
    'ngcf': NGCF,
    'neumf': NeuMF,
    'nfm': NFM,
    'multi-vae': VAE,
    'item2vec': Item2Vec,
}

initializer_param_config = {
    'normal': {'mean':0.0, 'std':0.01},
    'uniform': {'a':0.0, 'b':1.0},
    'xavier_normal': {'gain':1.0},
    'xavier_uniform': {'gain':1.0}
}

initializer_config = {
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_
}

def init_seed(seed, reproducibility):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn

    Parameters
    ----------
    seed : int
        random seed
    reproducibility : bool
        Whether to require reproducibility
    '''
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
