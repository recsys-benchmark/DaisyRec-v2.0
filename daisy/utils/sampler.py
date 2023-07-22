import numpy as np
from daisy.utils.utils import get_random_choice_custom
from torch.utils.data import DataLoader


class AbstractSampler(object):
    def __init__(self, config):
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.item_num = config['item_num']
        self.ur = config['train_ur']

    def sampling(self):
        raise NotImplementedError


class BasicNegtiveSampler(AbstractSampler):
    def __init__(self, df, config):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
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
        super(BasicNegtiveSampler, self).__init__(config)
        self.user_num = config['user_num']
        self.num_ng = config['num_ng']
        self.inter_name = config['INTER_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']
        self.sampling_batch = config['batch_size']
        self.loss_type = config['loss_type'].upper()

        assert self.sample_method in [
            'uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None

        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError(
                    'loss function (BPR, TL, HL) need num_ng > 0')

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
        else:  # uniform
            # all negative samples are sampled by uniform distribution
            for u in range(self.user_num):
                past_inter = list(self.ur[u])
                js[u] = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter),
                    size=self.num_ng
                )

        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name,
                                 self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name,
                                 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name,
                               'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError

    def set_diff_sampling(self):

        if self.sample_method != "uniform":
            raise NotImplementedError(
                "popularity batch sampling not implemeneted")

        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError(
                    'loss function (BPR, TL, HL) need num_ng > 0')

        df_len = len(self.df)
        neg_samples = np.ndarray((df_len, self.num_ng))
        all_users = self.df[self.uid_name]

        for i in range(df_len):
            cand_neg_i = self.ur[~all_users[i]]
            neg_samples[i] = get_random_choice_custom(cand_neg_i, self.num_ng)

        # Assign to dataframe
        self.df['neg_set'] = list(neg_samples)

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name,
                                 self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name,
                                 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name,
                               'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError

    def guess_and_check_sampling(self):
        '''
        Finds num_ng negative items *for every* user-item pair
        Simply guesses num_ng items and checks if user has interacted with the guesses

        If popularity sampling employed, the set difference between all items and user-interacted items
        will be taken and sampled from
        '''
        # Handle 0 negative sample case
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError(
                    'loss function (BPR, TL, HL) need num_ng > 0')

        # This function checks if the guesses for a given user are correct
        def uniform_items_check(num_uniform_samples, all_items, past_interactions):

            neg_items = set()
            interaction_ratio = self.item_num // len(past_interactions)

            if True or interaction_ratio > 1:

                # Repeat for every uniform sample needed
                for _ in range(num_uniform_samples):

                    # Guess an item
                    new_neg_item = np.random.randint(self.item_num)

                    # While guess is already interacted or guess already in negative set, guess again
                    while new_neg_item in past_interactions or new_neg_item in neg_items:
                        new_neg_item = np.random.randint(self.item_num)

                    neg_items.add(new_neg_item)

                return list(neg_items)

            else:
                '''
                This code is currently unreachable. 
                Used for when guess-and-check is not as efficient as set difference.
                Further exploration is needed on when this needs to be the case. 
                My hypothesis is when interaction ratio > 1, but empirical testing shows otherwise
                Also using floor division for calculating interaction ratio for some performance gains
                '''
                un_interacted_items = all_items.difference(past_interactions)
                neg_items = np.random.choice(
                    list(un_interacted_items), size=num_uniform_samples)
                return neg_items

        def combined_sampler(num_uniform_samples, num_other_samples=0):
            '''
            One-time use function for getting all samples
            Will perform guess-and-check for uniform samples, set difference for popularity samples
            '''

            # Guess the items
            candidate_negative_items = np.random.randint(
                self.item_num, size=(len(self.df), num_uniform_samples))

            # This is the final array shape if there are popularity (non-uniform) samples
            result_items = np.ndarray(
                (len(self.df), num_uniform_samples+num_other_samples))

            # Get all user ids (in order) and item ids
            all_user_ids = self.df[self.uid_name]
            all_item_ids_set = set(range(self.item_num))
            all_item_ids_arr = np.arange(self.item_num)

            # Run through all the guessed arrays, check them
            for index, candidate_items_list in enumerate(candidate_negative_items):
                user_id = all_user_ids.iloc[index]
                past_interactions = self.ur[user_id]
                if not len(past_interactions):
                    print(
                        f"index:{index} user:{user_id} \nrow:\n {self.df.iloc[index]}\n")
                # When we convert the items to set, due to duplicates the length of the set might decrease
                candidate_items_set = set(candidate_items_list)

                # If check failed: there are duplicates in the array or item already interacted with
                if len(candidate_items_set) != num_uniform_samples or past_interactions.intersection(candidate_items_set):

                    # Guess and check a fresh set and re-assign
                    candidate_items_list = uniform_items_check(
                        num_uniform_samples, all_item_ids_set, past_interactions)
                    candidate_negative_items[index] = candidate_items_list

                # If there are popularity samples then re-assign
                if num_other_samples:
                    other_negs = np.random.choice(
                        all_item_ids_arr,
                        size=num_other_samples,
                        p=self.pop_prob
                    )
                    result_items[index] = np.concatenate(
                        (candidate_items_list, other_negs), axis=None)

            return list(result_items) if num_other_samples else list(candidate_negative_items)

        # Peform explosion and conversion
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num
            self.df['neg_set'] = combined_sampler(uniform_num, other_num)
        else:
            self.df['neg_set'] = combined_sampler(self.num_ng)

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name,
                                 self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name,
                                 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name,
                               'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError

    def batch_sampling(self, sampling_batch_size=None):
        '''
        Finds num_ng negative items *for every* user-item pair
        Samples from items within batches of specified batch size

        popularity sampling not implemented
        '''
        # We will implement non-uniform later
        if self.sample_method != "uniform":
            raise NotImplementedError(
                "popularity batch sampling not implemeneted")

        # Handle case where no negative sampling is needed
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda _: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError(
                    'loss function (BPR, TL, HL) need num_ng > 0')

        # Sampling batch size is same as training batch by default
        if not sampling_batch_size:
            sampling_batch_size = self.sampling_batch

        # Shuffle dataframe and initialise negative set
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df['neg_set'] = np.NaN
        # So arrays can be a assigned to a single cell in df
        self.df['neg_set'] = self.df['neg_set'].astype('object')

        # Python generator that splits the dataset into batches
        def batch_generator():
            batch_index_start = 0
            batch_index_end = sampling_batch_size
            df_length = len(self.df)

            while batch_index_end < df_length:
                next_batch = self.df.iloc[batch_index_start:batch_index_end]
                yield (batch_index_start, batch_index_end), next_batch
                batch_index_start += sampling_batch_size
                batch_index_end += sampling_batch_size

            last_batch = self.df.iloc[batch_index_start:df_length]
            yield (batch_index_start, df_length), last_batch

        # Guesses an item from the entire database and checks
        def guess_and_check(past_interactions):
            neg_items = set()

            for _ in range(self.num_ng):
                new_neg_item = np.random.randint(self.item_num)
                while new_neg_item in past_interactions or new_neg_item in neg_items:
                    new_neg_item = np.random.randint(self.item_num)
                neg_items.add(new_neg_item)

            return np.array(list(neg_items))

        # We create a function to apply to every row in the df
        def get_neg_sample(user_id, batch_items_set):

            # For every row in the batch we will perform set difference
            past_interactions = self.ur[user_id]
            set_difference = batch_items_set.difference(past_interactions)

            # If the set difference returns enough samples, randomly choose
            if len(set_difference) >= self.num_ng:
                return np.random.choice(list(set_difference), size=self.num_ng)

            # Else perform guess-and-check sampling
            return guess_and_check(past_interactions)

        all_batches = batch_generator()
        for indices, batch in all_batches:
            start_i, end_i = indices

            # Next we create a list of all items available in the batches
            batch_items = batch[self.iid_name].unique()
            batch_items_set = set(batch_items)

            # Get all the users in the batch in order
            batch_users = batch[self.uid_name]

            # Append the negative sample across all given indices
            for i in range(start_i, end_i):
                self.df.at[i, 'neg_set'] = get_neg_sample(
                    batch_users[i], batch_items_set)
                # self.df.at[i, 'neg_set'] = guess_and_check(self.ur[batch_users[i]])

        # Perform explosion and conversion
        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name,
                                 self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name,
                                 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name,
                               'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError(
                "That loss type has not yet been implemented")


class ParallelNegativeSampler(AbstractSampler):

    def __init__(self, df, config):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
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
        super(ParallelNegativeSampler, self).__init__(config)
        self.user_num = config['user_num']
        self.num_ng = config['num_ng']
        self.inter_name = config['INTER_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']
        self.sampling_batch = config['batch_size']
        self.loss_type = config['loss_type'].upper()

        assert self.sample_method in [
            'uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None

        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def guess_and_check_negative_sampler(batch):
        pass

    def popularity_sampler(batch):
        pass

    def sampling(self, batch_size, num_workers=4):

        sampler_data_loader = None

        if self.sample_method == 'uniform':
            sampler_data_loader = DataLoader(
                self.df.to_numpy(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=self.guess_and_check_negative_sampler
            )
        else:
            sampler_data_loader = DataLoader(
                self.df.to_numpy(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=self.popularity_sampler
            )


class SkipGramNegativeSampler(AbstractSampler):
    def __init__(self, df, config, discard=False):
        '''
        skip-gram negative sampling class for <target_i, context_i, label>

        Parameters
        ----------
        df : pd.DataFrame
            training set
        rho : float, optional
            threshold to discard word in a sequence, by default 1e-5
        context_window: int, context range around target
        train_ur: dict, ground truth for each user in train set
        item_num: int, the number of items
        '''
        super(SkipGramNegativeSampler, self).__init__(config)
        self.context_window = config['context_window']

        word_frequecy = df[self.iid_name].value_counts()
        prob_discard = 1 - np.sqrt(config['rho'] / word_frequecy)

        if discard:
            rnd_p = np.random.uniform(low=0., high=1., size=len(df))
            discard_p_per_item = df[self.iid_name].map(prob_discard).values
            df = df[rnd_p >= discard_p_per_item]

        self.train_seqs = self._build_seqs(df)

    def sampling(self):
        sgns_samples = []

        for u, seq in self.train_seqs.iteritems():
            past_inter = list(self.ur[u])
            cands = np.setdiff1d(np.arange(self.item_num), past_inter)

            for i in range(len(seq)):
                target = seq[i]
                # generate positive sample
                context_list = []
                j = i - self.context_window
                while j <= i + self.context_window and j < len(seq):
                    if j >= 0 and j != i:
                        context_list.append(seq[j])
                        sgns_samples.append([target, seq[j], 1])
                    j += 1
                # generate negative sample
                num_ng = len(context_list)
                for neg_item in np.random.choice(cands, size=num_ng):
                    sgns_samples.append([target, neg_item, 0])

        return np.array(sgns_samples)

    def _build_seqs(self, df):
        train_seqs = df.groupby(self.uid_name)[self.iid_name].agg(list)

        return train_seqs
