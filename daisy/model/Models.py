from daisy.model.AbstractRecommender import GeneralRecommender

def RecommenderModel(algo_name: str) -> GeneralRecommender:
    """Takes the model name as a string and returns the model class
    """
    if algo_name == 'mf':
        from daisy.model.accuracyRecommender.MFRecommender import MF
        return MF
    elif algo_name == 'fm':
        from daisy.model.accuracyRecommender.FMRecommender import FM
        return FM
    elif algo_name == 'nfm':
        from daisy.model.accuracyRecommender.NFMRecommender import NFM
        return NFM
    elif algo_name == 'ngcf':
        from daisy.model.accuracyRecommender.NGCFRecommender import NGCF
        return NGCF
    elif algo_name == 'ease':
        from daisy.model.accuracyRecommender.EASERecommender import EASE
        return EASE
    elif algo_name == 'slim':
        from daisy.model.accuracyRecommender.SLiMRecommender import SLiM
        return SLiM
    elif algo_name == 'multi-vae':
        from daisy.model.accuracyRecommender.VAECFRecommender import VAECF
        return VAECF
    elif algo_name == 'neumf':
        from daisy.model.accuracyRecommender.NeuMFRecommender import NeuMF
        return NeuMF
    elif algo_name == 'mostpop':
        from daisy.model.accuracyRecommender.PopRecommender import MostPop
        return MostPop
    elif algo_name == 'itemknn':
        from daisy.model.accuracyRecommender.KNNCFRecommender import ItemKNNCF
        return ItemKNNCF
    elif algo_name == 'puresvd':
        from daisy.model.accuracyRecommender.PureSVDRecommender import PureSVD
        return PureSVD
    elif algo_name == 'item2vec':
        from daisy.model.accuracyRecommender.Item2VecRecommender import Item2Vec
        return Item2Vec
    elif algo_name == 'lightgcn':
        from daisy.model.accuracyRecommender.LightGCNRecommender import LightGCN
        return LightGCN
    elif algo_name == 'sndmostpop':
        from daisy.model.accuracyRecommender.SecondMostPopRecommender import SndMostPop
        return SndMostPop
    else:
        raise ModuleNotFoundError(f"Model name '{algo_name}' not found")



    

