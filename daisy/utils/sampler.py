import numpy as np
import scipy.sparse as sp

class Sampler(object):
    def __init__(self, df, ur, user_num, item_num, config):
        """
        negative sampling class for <u, i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, No. of nagative sampling per sample, default is 4
        sample_method : str, sampling method, default is 'uniform',
                        'uniform' discrete uniform sampling
                        'high-pop' sample items with high popularity as priority
                        'low-pop' sample items with low popularity as prority
        sample_ratio : float, scope [0, 1], it determines the ratio that the other sample method except 'uniform' occupied, default is 0
        """
        self.user_num = user_num
        self.item_num = item_num
        self.num_ng = config['num_ng']
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']

        assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.ur = ur
        self.pop_prob = None
        
        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            return self.df

        js = np.zeros((self.user_num, self.num_ng), dtype=np.int32)
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            for u in range(self.user_num):
                past_inter = list(self.ur[u])

                uni_negs = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter), 
                    size=uniform_num
                )
                other_negs = np.random.choice(
                    np.arange(self.item_num),
                    size=other_num,
                    p=self.pop_prob
                )
                js[u] = np.concatenate((uni_negs, other_negs), axis=None)

        else:
            # all negative samples are sampled by uniform distribution
            for u in range(self.user_num):
                # sample = np.random.choice(self.item_num, size=)
                past_inter = list(self.ur[u])
                js[u] = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter), 
                    size=self.num_ng
                )

        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])

        return self.df
