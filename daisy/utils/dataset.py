import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# def guess_and_check_negative_sampler(train_ur, data):
    
#     for user, item in data:
#         print(user, item)

# def collate_fn_wrapper(data):

#     def collate_fn(data):
#         return guess_and_check_negative_sampler(data, train_ur)
    
#     return collate_fn

# def myfun():
#     def inner(data):
#         print(data)
    
#     return inner



class BasicDataset(Dataset):
    def __init__(self, samples, config):
        '''
        convert array-like <u, i, j> / <u, i, r> / <target_i, context_i, label>

        Parameters
        ----------
        samples : np.array
            samples generated by sampler
        '''        
        super(BasicDataset, self).__init__()
        self.data = samples
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def guess_and_check_negative_sampler(self, data):

        num_negatives_per_ui_pair = self.config['num_ng']
        user_items_interaction_map = self.config['train_ur']
        batch_size = self.config['batch_size']
        num_of_items_total = self.config['item_num']

        # we need to initialise the resulting array: batch * num_negatives rows, 3 columns for each <user, positive-item, negative-item>
        final_train_data = np.ndarray((batch_size * num_negatives_per_ui_pair, 3))

        # first we go through the data. We repeat each sample num_negative times for the sampling
        for index, row in enumerate(np.repeat(data, repeats=num_negatives_per_ui_pair, axis=0)):
            user, item = row

            # guess an item
            negative_item = np.random.randint(num_of_items_total)

            past_interactions = user_items_interaction_map[user]

            # check the item. Guess again while in past interactions
            while negative_item in past_interactions:
                negative_item = np.random.randint(num_of_items_total)

            # if all is well, add it in to the list
            final_train_data[index] = [user, item, negative_item]

        return torch.tensor(final_train_data.transpose(), dtype=torch.long)
            
                



    def get_dataloader(self, batch_size, shuffle, num_workers=4):


        return DataLoader(
            self.data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=self.guess_and_check_negative_sampler
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
