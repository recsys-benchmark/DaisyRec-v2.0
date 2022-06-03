import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from scipy.sparse.linalg import spsolve


class WRMF(object):
    def __init__(self, 
                 user_num, 
                 item_num, 
                 df, 
                 factors=20,
                 epochs=10, 
                 alpha=40,
                 reg_2=0.1,  
                 seed=2019):
        """
        WRMF Recommender Class
        Parameters
        ----------
        user_num : int, number of users
        item_num : int, number of items
        df : pd.DataFrame, rating data frame
        factors : latent factor number
        epochs : number of training epochs
        alpha : amplify coefficients
        reg_2 : float, second-order regularization term
        seed : random seed
        """
        train_set = self._convert_df(user_num, item_num, df)

        self.epochs = epochs
        self.rstate = np.random.RandomState(seed)
        self.C = alpha * train_set
        self.user_num, self.item_num = user_num, item_num

        self.X = sp.csr_matrix(self.rstate.normal(scale=0.01, 
                                                  size=(user_num, factors)))
        self.Y = sp.csr_matrix(self.rstate.normal(scale=0.01, 
                                                  size=(item_num, factors)))
        self.X_eye = sp.eye(user_num)
        self.Y_eye = sp.eye(item_num)
        self.lambda_eye = reg_2 * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def fit(self):
        for _ in tqdm(range(self.epochs)):
            yTy = self.Y.T.dot(self.Y)
            xTx = self.X.T.dot(self.X)
            for u in range(self.user_num):
                Cu = self.C[u, :].toarray()
                Pu = Cu.copy()
                Pu[Pu != 0] = 1
                CuI = sp.diags(Cu, [0])
                yTCuIY = self.Y.T.dot(CuI).dot(self.Y)
                yTCuPu = self.Y.T.dot(CuI + self.Y_eye).dot(Pu.T)
                self.X[u] = spsolve(yTy + yTCuIY + self.lambda_eye, yTCuPu)
            for i in range(self.item_num):
                Ci = self.C[:, i].T.toarray()
                Pi = Ci.copy()
                Pi[Pi != 0] = 1
                CiI = sp.diags(Ci, [0])
                xTCiIX = self.X.T.dot(CiI).dot(self.X)
                xTCiPi = self.X.T.dot(CiI + self.X_eye).dot(Pi.T)
                self.Y[i] = spsolve(xTx + xTCiIX + self.lambda_eye, xTCiPi)

        self.user_vec, self.item_vec = self.X, self.Y.T

        # complete prediction matrix in fit process and save time for get rank list
        pred_mat = self.user_vec.dot(self.item_vec)
        self.pred_mat = pred_mat.A

    def predict(self, u, i):
        prediction = self.pred_mat[u, i]
        return prediction

    def _convert_df(self, user_num, item_num, df):
        '''Process Data to make WRMF available'''
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csr_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

    
"""
Reference: Yifan Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" in ICDM 2008.
@author: wubin
"""
import numpy as np
from time import time
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
from util import timer, tool


class WRMF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(WRMF, self).__init__(dataset, conf)
        self.embedding_size = conf["embedding_size"]
        self.alpha = conf["alpha"]
        self.topK = conf["topk"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess
        self.Cui = np.zeros(shape=[self.num_users, self.num_items], dtype=np.float32)
        self.Pui = np.zeros(shape=[self.num_users, self.num_items], dtype=np.float32)
        for u in np.arange(self.num_users):
            items_by_user_id = self.dataset.train_matrix[u].indices
            for i in items_by_user_id:
                self.Cui[u, i] = self.alpha
                self.Pui[u, i] = 1.0
        self.lambda_eye = self.reg_mf * tf.eye(self.embedding_size)
    
    def _create_placeholders(self):
        self.user_id = tf.placeholder(tf.int32, [1])
        self.Cu = tf.placeholder(tf.float32, [self.num_items, 1])  
        self.Pu = tf.placeholder(tf.float32, [self.num_items, 1])

        self.item_id = tf.placeholder(tf.int32, [1])
        self.Ci = tf.placeholder(tf.float32, [self.num_users, 1])  
        self.Pi = tf.placeholder(tf.float32, [self.num_users, 1])
    
    def _create_variables(self):
        initializer = tool.get_initializer(self.init_method, self.stddev)
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]), name='user_embeddings', dtype=tf.float32)
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]), name='item_embeddings', dtype=tf.float32)
        
    def _create_optimizer(self):    
        YTY = tf.matmul(self.item_embeddings, self.item_embeddings, transpose_a=True)
        YTCuIY = tf.matmul(self.item_embeddings, tf.multiply(self.Cu, self.item_embeddings), transpose_a=True)  
        YTCupu = tf.matmul(self.item_embeddings, tf.multiply(self.Cu+1, self.Pu), transpose_a=True)
        xu = tf.linalg.solve(YTY + YTCuIY + self.lambda_eye, YTCupu)
        self.update_user = tf.scatter_update(self.user_embeddings, self.user_id, tf.transpose(xu))

        XTX = tf.matmul(self.user_embeddings, self.user_embeddings, transpose_a=True)
        XTCIIX = tf.matmul(self.user_embeddings, tf.multiply(self.Ci, self.user_embeddings), transpose_a=True)  
        XTCIpi = tf.matmul(self.user_embeddings, tf.multiply(self.Ci+1, self.Pi), transpose_a=True)
        xi = tf.linalg.solve(XTX + XTCIIX + self.lambda_eye, XTCIpi)
        self.update_item = tf.scatter_update(self.item_embeddings, self.item_id, tf.transpose(xi))
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_optimizer()
        
    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1, self.num_epochs+1):
            training_start_time = time()
            print('solving for user vectors...')
            for user_id in range(self.num_users):
                feed = {self.user_id: [user_id],
                        self.Pu: self.Pui[user_id].T.reshape([-1, 1]),
                        self.Cu: self.Cui[user_id].T.reshape([-1, 1])}
                self.sess.run(self.update_user, feed_dict=feed)

            print('solving for item vectors...')
            for item_id in range(self.num_items):
                feed = {self.item_id: [item_id],
                        self.Pi: self.Pui[:,item_id].reshape([-1, 1]),
                        self.Ci: self.Cui[:,item_id].reshape([-1, 1])}
                self.sess.run(self.update_item, feed_dict=feed)
           
            self.logger.info('iteration %i finished in %f seconds' % (epoch, time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
                
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.user_embeddings, self.item_embeddings])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        if candidate_items_userids is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            for user_id, items_by_user_id in zip(user_ids, candidate_items_userids):
                user_embed = self._cur_user_embeddings[user_id]
                items_embed = self._cur_item_embeddings[items_by_user_id]
                ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))
        return ratings
