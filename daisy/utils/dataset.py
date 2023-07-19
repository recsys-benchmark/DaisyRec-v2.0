import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, df, config):
        '''
        convert array-like <u, i, j> / <u, i, r> / <target_i, context_i, label>

        Parameters
        ----------
        df : np.array
        training dataset
        '''
        super(BasicDataset, self).__init__()
        self.data = df
        self.config = config
        self.sample_method = config['sample_method']

        # if sampling is non-uniform, initialise the popularity score of each item
        if self.sample_method != 'uniform':
            itemIDName = config['IID_NAME']

            # get the counts of all items in the dataframe
            pop = df.groupby(itemIDName).size()
            # rescale to [0, 1]
            pop /= pop.sum()

            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(config['item_num'])
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(config['item_num'])
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def popularity_sampler(self, data):
        raise KeyboardInterrupt

    def guess_and_check_negative_sampler(self, data):
        num_negatives_per_ui_pair = self.config['num_ng']
        user_items_interaction_map = self.config['train_ur']
        batch_size = self.config['batch_size']
        num_of_items_total = self.config['item_num']

        # we need to initialise the resulting array: batch * num_negatives rows, 3 columns for each <user, positive-item, negative-item>
        final_train_data = np.ndarray(
            (batch_size * num_negatives_per_ui_pair, 3))

        # first we go through the data. We repeat each sample num_negative times for the sampling
        for index, row in enumerate(np.repeat(data, repeats=num_negatives_per_ui_pair, axis=0)):
            user, item = row

            # guess an item
            negative_item = np.random.randint(num_of_items_total)

            # check the item. Guess again while in past interactions
            while user_items_interaction_map[user, negative_item]:
                negative_item = np.random.randint(num_of_items_total)

            # if all is well, add it in to the list
            final_train_data[index] = [user, item, negative_item]

        return torch.tensor(final_train_data.transpose(), dtype=torch.long)

    def get_dataloader(self, batch_size=256, shuffle=True, num_workers=4):

        if self.sample_method == 'uniform':
            return DataLoader(
                self.data.to_numpy(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=self.guess_and_check_negative_sampler
            )
        else:
            return DataLoader(
                self.data.to_numpy(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=self.popularity_sampler
            )


class CandidatesDataset(Dataset):
    def __init__(self, ucands):
        super(CandidatesDataset, self).__init__()
        self.data = ucands

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index][0]), torch.tensor(self.data[index][1])


class AEDataset(Dataset):
    def __init__(self, train_set, yield_col='user'):
        """
        covert user in train_set to array-like <u> / <i> for AutoEncoder-like algorithms
        Parameters
        ----------
        train_set : pd.DataFrame
            training set
        yield_col : string
            column name used to generate array
        """
        super(AEDataset, self).__init__()
        self.data = train_set[yield_col].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
