from .AbstractRecommender import GeneralRecommender

def RecommenderModel(algo_name: str) -> GeneralRecommender:
    """Takes the model name as a string and returns the model class
    """
    if algo_name == 'mf':
        from .MFRecommender import MF
        return MF
    elif algo_name == 'fm':
        from .FMRecommender import FM
        return FM
    elif algo_name == 'nfm':
        from .NFMRecommender import NFM
        return NFM
    elif algo_name == 'ngcf':
        from .NGCFRecommender import NGCF
        return NGCF
    elif algo_name == 'ease':
        from .EASERecommender import EASE
        return EASE
    elif algo_name == 'slim':
        from .SLiMRecommender import SLiM
        return SLiM
    elif algo_name == 'multi-vae':
        from .VAECFRecommender import VAECF
        return VAECF
    elif algo_name == 'neumf':
        from .NeuMFRecommender import NeuMF
        return NeuMF
    elif algo_name == 'mostpop':
        from .PopRecommender import MostPop
        return MostPop
    elif algo_name == 'itemknn':
        from .KNNCFRecommender import ItemKNNCF
        return ItemKNNCF
    elif algo_name == 'puresvd':
        from .PureSVDRecommender import PureSVD
        return PureSVD
    elif algo_name == 'item2vec':
        from .Item2VecRecommender import Item2Vec
        return Item2Vec
    elif algo_name == 'lightgcn':
        from .LightGCNRecommender import LightGCN
        return LightGCN
    elif algo_name == 'sndmostpop':
        from .SecondMostPopRecommender import SndMostPopRecommender
        return SndMostPopRecommender
    else:
        raise ModuleNotFoundError(f"Model name '{algo_name}' not found")



    

