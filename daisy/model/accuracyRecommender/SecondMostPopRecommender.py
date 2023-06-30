from daisy.model.AbstractRecommender import GeneralRecommender
from pandas import DataFrame
import numpy as np
import torch


class SndMostPop(GeneralRecommender):
    '''
    Model recommends user the second most popular item in the list
    '''

    def __init__(self, config):
        super(SndMostPop, self).__init__(config)
        self.config = config

    def fit(self, training_df: DataFrame) -> np.ndarray:
        '''
        Ranks for item in the training dataframe by its popularity (number of user-item interactions)
        returns this 1-D array of ranks
        '''
        items_column = training_df[self.config['IID_NAME']]  # config['IID_NAME'] usually returns 'item'
        self.item_counts_series = items_column.value_counts()

        return self.item_counts_series

    def rank(self, test_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        '''
        Returns rank of all 

        '''

        # Pick out the top k
        k = self.config['topk']

        predictions = np.zeros((len(test_loader.dataset), k))
        predictions_cur_index = 0

        for batch in test_loader:
            _, candidate_items_lists = batch
            # we get 128 (u, [1000 item]) pairs
            for candidate_items in candidate_items_lists:

                # convert to np array
                candidate_items = np.array(candidate_items)
                self.item_counts_array = np.zeros(self.config['item_num'])
                self.item_counts_array[self.item_counts_series.index] = self.item_counts_series.values

                # we must first score the items
                candidate_item_scores = self.item_counts_array[candidate_items]

                # Find the indices of the top k scores
                top_k_indices = np.argsort(candidate_item_scores)[-k:]

                # Use the indices to get the item IDs with the top k scores
                top_k_items = candidate_items[top_k_indices]

                # Put into predictons
                predictions[predictions_cur_index] = top_k_items
                predictions_cur_index += 1

        return predictions

    def predict(self, u, i):
        try:
            return self.item_scores[i]
        except AttributeError:
            raise RuntimeError("Fit the model before trying to predict")
