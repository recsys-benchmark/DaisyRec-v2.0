from AbstractRecommender import GeneralRecommender

def RecommenderModel(model_name: str) -> GeneralRecommender:
    """Takes the model name as a string and returns the model class
    """
    if model_name == 'mf':
        from daisy.model.MFRecommender import MF
        return MF
    elif model_name == 'fm':
        from daisy.model.FMRecommender import FM
        return FM
    elif model_name == 'nfm':
        from daisy.model.NFMRecommender import NFM
        return NFM
    elif model_name == 'ngcf':
        from daisy.model.NGCFRecommender import NGCF
        return NGCF
    elif model_name == 'ease':
        from daisy.model.EASERecommender import EASE
        return EASE
    elif model_name == 'slim':
        from daisy.model.SLiMRecommender import SLiM
        return SLiM
    elif model_name == 'multi-vae':
        from daisy.model.VAECFRecommender import VAECF
        return VAECF
    elif model_name == 'neumf':
        from daisy.model.NeuMFRecommender import NeuMF
        return NeuMF
    elif model_name == 'mostpop':
        from daisy.model.PopRecommender import MostPop
        return MostPop
    elif model_name == 'itemknn':
        from daisy.model.KNNCFRecommender import ItemKNNCF
        return ItemKNNCF
    elif model_name == 'puresvd':
        from daisy.model.PureSVDRecommender import PureSVD
        return PureSVD
    elif model_name == 'item2vec':
        from daisy.model.Item2VecRecommender import Item2Vec
        return Item2Vec
    elif model_name == 'lightgcn':
        from daisy.model.LightGCNRecommender import LightGCN
        return LightGCN
    else:
        raise ModuleNotFoundError(f"Model name '{model_name}' not found")



    

