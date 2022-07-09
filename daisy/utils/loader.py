import os
import gc
from pickle import FALSE
import re
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from collections import defaultdict, Counter

class Interactions(object):
    def __init__(self, config):
        """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        user_num : int, the number of users
        item_num : int, the number of items
        """
        self.src = config['dataset']
        self.prepro = config['prepro']
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.tid_name = config['TID_NAME']
        self.inter_name = config['INTER_NAME']
        self.binary = config['binary_inter']
        self.pos_threshold = config['positive_threshold']
        self.level = config['level'] # ui, u, i

    def get_data(self):
        df = self.__get_raw_data()
        df = self.__reserve_pos(df)
        df = self.__binary_inter(df)
        df = self.__core_filter(df)
        self.user_num, self.item_num = self.__get_stats(df)
        df = self.__category_encoding(df)
        print(f'Finish loading [{self.src}]-[{self.prepro}] dataset')

        return df

    def __category_encoding(self, df):
        # encoding user_id and item_id
        self.uid_token = pd.Categorical(df['user']).categories.to_numpy()
        self.iid_token = pd.Categorical(df['item']).categories.to_numpy()
        self.token_uid = {uid: token for token, uid in enumerate(self.uid_token)}
        self.token_iid = {iid: token for token, iid in enumerate(self.iid_token)}
        df['user'] = pd.Categorical(df['user']).codes
        df['item'] = pd.Categorical(df['item']).codes

        return df

    def __get_stats(self, df):
        user_num = df['user'].nunique()
        item_num = df['item'].nunique()

        return user_num, item_num

    def __get_illegal_ids_by_inter_num(self, df, field, inter_num, min_num):
        ids = set()
        for id_ in df[field].values:
            if inter_num[id_] < min_num:
                ids.add(id_)
        return ids

    def __core_filter(self, df):
        # which type of pre-dataset will use
        if self.prepro == 'origin':
            pass
        elif self.prepro.endswith('filter'):
            pattern = re.compile(r'\d+')
            filter_num = int(pattern.findall(self.prepro)[0])

            tmp1 = df.groupby(['user'], as_index=False)['item'].count()
            tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
            tmp2 = df.groupby(['item'], as_index=False)['user'].count()
            tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
            if self.level == 'ui':    
                df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'u':
                df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'i':
                df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
            del tmp1, tmp2
            gc.collect()

        elif self.prepro.endswith('core'):
            pattern = re.compile(r'\d+')
            core_num = int(pattern.findall(self.prepro)[0])

            if self.level == 'ui':
                user_inter_num = Counter(df[self.uid_name].values)
                item_inter_num = Counter(df[self.iid_name].values)
                while True:
                    ban_users = self.__get_illegal_ids_by_inter_num(df, 'user', user_inter_num, core_num)
                    ban_items = self.__get_illegal_ids_by_inter_num(df, 'item', item_inter_num, core_num)

                    if len(ban_users) == 0 and len(ban_items) == 0:
                        break

                    dropped_inter = pd.Series(False, index=df.index)
                    user_inter = df[self.uid_name]
                    item_inter = df[self.iid_name]
                    dropped_inter |= user_inter.isin(ban_users)
                    dropped_inter |= item_inter.isin(ban_items)
                    
                    user_inter_num -= Counter(user_inter[dropped_inter].values)
                    item_inter_num -= Counter(item_inter[dropped_inter].values)

                    dropped_index = df.index[dropped_inter]
                    df.drop(dropped_index, inplace=True)

            elif self.level == 'u':
                tmp = df.groupby(['user'], as_index=False)['item'].count()
                tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
                df = df.merge(tmp, on=['user'])
                df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_item'], axis=1, inplace=True)
            elif self.level == 'i':
                tmp = df.groupby(['item'], as_index=False)['user'].count()
                tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
                df = df.merge(tmp, on=['item'])
                df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_user'], axis=1, inplace=True)
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            gc.collect()

        else:
            raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')
        
        df = df.reset_index(drop=True)

        return df

    def __reserve_pos(self, df):
        # set rating >= threshold as positive samples
        if self.pos_threshold is not None:
            df = df.query(f'rating >= {self.pos_threshold}').reset_index(drop=True)
        return df

    def __binary_inter(self, df):
        # reset rating to interaction, here just treat all rating as 1
        if self.binary:
            df['rating'] = 1.0
        return df

    def __get_raw_data(self):
        df = pd.DataFrame()
        if self.src == 'ml-100k':
            df = pd.read_csv(f'./data/{self.src}/u.data', sep='\t', header=None,
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name], engine='python')

        elif self.src == 'ml-1m':
            df = pd.read_csv(f'./data/{self.src}/ratings.dat', sep='::', header=None, 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name], engine='python')
        
        elif self.src == 'ml-10m':
            df = pd.read_csv(f'./data/{self.src}/ratings.dat', sep='::', header=None, 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name], engine='python')
        elif self.src == 'ml-20m':
            df = pd.read_csv(f'./data/{self.src}/ratings.csv')
            df.rename(columns={'userId':self.uid_name, 'movieId': self.iid_name}, inplace=True)

        elif self.src == 'netflix':
            cnt = 0
            tmp_file = open(f'./data/{self.src}/training_data.csv', 'w')
            tmp_file.write(f'{self.uid_name},{self.iid_name},{self.inter_name},{self.tid_name}' + '\n')
            for f in os.listdir(f'./data/{self.src}/training_set/'):
                cnt += 1
                if cnt % 5000 == 0:
                    print(f'Finish Process {cnt} file......')
                txt_file = open(f'./data/{self.src}/training_set/{f}', 'r')
                contents = txt_file.readlines()
                item = contents[0].strip().split(':')[0]
                for val in contents[1:]:
                    user, rating, timestamp = val.strip().split(',')
                    tmp_file.write(','.join([user, item, rating, timestamp]) + '\n')
                txt_file.close()
            tmp_file.close()

            df = pd.read_csv(f'./data/{self.src}/training_data.csv')
            df[self.inter_name] = df.rating.astype(float)
            df[self.tid_name] = pd.to_datetime(df['timestamp'])

        elif self.src == 'lastfm':
            df = pd.read_csv(f'./data/{self.src}/user_artists.dat', sep='\t')
            df.rename(columns={'userID': self.uid_name, 'artistID': self.iid_name, 'weight': self.inter_name}, inplace=True)
            # treat weight as interaction, as 1
            df[self.inter_name] = 1.0
            # fake timestamp column
            df[self.tid_name] = 1

        elif self.src == 'book-x':
            df = pd.read_csv(f'./data/{self.src}/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
            df.rename(columns={'User-ID': self.uid_name, 'ISBN': self.iid_name, 'Book-Rating': self.inter_name}, inplace=True)
            # fake timestamp column
            df[self.tid_name] = 1

        elif self.src == 'pinterest':
            # TODO this dataset has wrong source URL, we will figure out in future
            pass

        elif self.src == 'amazon-cloth':
            df = pd.read_csv(f'./data/{self.src}/ratings_Clothing_Shoes_and_Jewelry.csv', 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])

        elif self.src == 'amazon-electronic':
            df = pd.read_csv(f'./data/{self.src}/ratings_Electronics.csv', 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])

        elif self.src == 'amazon-book':
            df = pd.read_csv(f'./data/{self.src}/ratings_Books.csv', 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name], low_memory=False)
            df = df[df[self.tid_name].str.isnumeric()].copy()
            df[self.tid_name] = df[self.tid_name].astype(int)

        elif self.src == 'amazon-music':
            df = pd.read_csv(f'./data/{self.src}/ratings_Digital_Music.csv', 
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])

        elif self.src == 'epinions':
            d = sio.loadmat(f'./data/{self.src}/rating_with_timestamp.mat')
            prime = []
            for val in d['rating_with_timestamp']:
                user, item, rating, timestamp = val[0], val[1], val[3], val[5]
                prime.append([user, item, rating, timestamp])
            df = pd.DataFrame(prime, columns=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])
            del prime
            gc.collect()

        elif self.src == 'yelp':
            json_file_path = f'./data/{self.src}/yelp_academic_dataset_review.json'
            prime = []
            for line in open(json_file_path, 'r', encoding='UTF-8'):
                val = json.loads(line)
                prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
            df = pd.DataFrame(prime, columns=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])
            df[self.tid_name] = pd.to_datetime(df[self.tid_name])
            del prime
            gc.collect()

        elif self.src == 'citeulike':
            user = 0
            dt = []
            for line in open(f'./data/{self.src}/users.dat', 'r'):
                val = line.split()
                for item in val:
                    dt.append([user, item])
                user += 1
            df = pd.DataFrame(dt, columns=[self.uid_name, self.iid_name])
            # fake timestamp column
            df[self.tid_name] = 1
            df[self.inter_name] = 1.0

        else:
            raise NotImplementedError('Invalid Dataset Error')
        
        return df


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


def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=1000):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
    
    return test_ucands


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
