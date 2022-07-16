'''
@inproceedings{steck2019embarrassingly,
  title={Embarrassingly shallow autoencoders for sparse data},
  author={Steck, Harald},
  booktitle={The World Wide Web Conference},
  pages={3251--3257},
  year={2019}
}
'''
import numpy as np
import scipy.sparse as sp

from daisy.model.AbstractRecommender import GeneralRecommender

class EASE(GeneralRecommender):
    def __init__(self, config):
        super(EASE, self).__init__(config)
        self.inter_name = config['INTER_NAME']
        self.iid_name = config['IID_NAME']
        self.uid_name = config['UID_NAME']

        self.user_num = config['user_num']
        self.item_num = config['item_num']

        self.reg_weight = config['reg_weight']


    def fit(self, train_set):
        row_ids = train_set[self.uid_name].values
        col_ids = train_set[self.iid_name].values
        values = train_set[self.inter_name].values

        X = sp.csr_matrix((values, (row_ids, col_ids)), shape=(self.user_num, self.item_num)).astype(np.float32)


    def predict(self, u, i):
        pass

    def rank(self, test_loader):
        pass

    def full_rank(self, u):
        pass
