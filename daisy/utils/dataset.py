import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, df):
        '''
        convert array-like <u, i, j> / <u, i, r> / <target_i, context_i, label>

        Parameters
        ----------
        df : np.array
        training dataset
        '''
        super(BasicDataset, self).__init__()
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def get_dataloader(self, batch_size=256, shuffle=True, num_workers=4):
        return DataLoader(
            self.data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
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
