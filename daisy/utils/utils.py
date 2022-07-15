import os
import gc
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from daisy.utils.loader import load_rate, split_test

def generate_experiment_data(dataset, prepro, test_method):
    """
    method of generating dataset for reproducing paper KPI
    Parameters
    ----------
    dataset : str, dataset name, available options: 'netflix', 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'book-x',
                                                    'amazon-cloth', 'amazon-electronic', 'amazon-book', 'amazon-music',
                                                    'epinions', 'yelp', 'citeulike'
    prepro : str, way to pre-process data, available options: 'origin', '5filter', '10filter'
    test_method : str, way to get test dataset, available options: 'tsbr', 'rsbr', 'tloo', 'rloo'

    Returns
    -------

    """
    if not os.path.exists('./experiment_data/'):
        os.makedirs('./experiment_data/')
    print(f'start process {dataset} with {prepro} method...')
    df, user_num, item_num = load_rate(dataset, prepro, False)
    print(f'test method : {test_method}')
    train_set, test_set = split_test(df, test_method, .2)
    train_set.to_csv(f'./experiment_data/train_{dataset}_{prepro}_{test_method}.dat', index=False)
    test_set.to_csv(f'./experiment_data/test_{dataset}_{prepro}_{test_method}.dat', index=False)
    print('Finish save train and test set...')
    del train_set, test_set, df
    gc.collect()

def get_ur(df):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur


def get_ir(df):
    """
    Method of getting item-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir

def convert_npy_mat(user_num, item_num, df):
    """
    method of convert dataframe to numpy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for _, row in df.iterrows():
        u, i, r = row['user'], row['item'], row['rating']
        mat[int(u), int(i)] = float(r)
    return mat

def build_candidates_set(test_ur, train_ur, config, drop_past_inter=True):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_num : No. of all items
    cand_num : int, the number of candidates
    drop_past_inter : drop items already appeared in train set

    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    item_num = config['item_num']
    candidates_num = config['cand_num']

    test_ucands, test_u = [], []
    for u, r in test_ur.items():
        sample_num = candidates_num - len(r) if len(r) <= candidates_num else 0
        if sample_num == 0:
            samples = np.random.choice(list(r), candidates_num)
        else:
            pos_items = list(r) + list(train_ur[u]) if drop_past_inter else list(r)
            neg_items = np.setdiff1d(np.arange(item_num), pos_items)
            samples = np.random.choice(neg_items, size=sample_num)
            samples = np.concatenate((samples, list(r)), axis=None)

        test_ucands.append([u, samples])
        test_u.append(u)
    
    return test_u, test_ucands

def get_adj_mat(n_users, n_items):
    """
    method of get Adjacency matrix
    Parameters
    --------
    n_users : int, the number of users
    n_items : int, the number of items

    Returns
    -------
    adj_mat: adjacency matrix
    norm_adj_mat: normal adjacency matrix
    mean_adj_mat: mean adjacency matrix
    """
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape)

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)

        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        return temp

    norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = mean_adj_single(adj_mat)

    print('already normalize adjacency matrix')
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
