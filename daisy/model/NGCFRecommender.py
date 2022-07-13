import torch
import torch.nn as nn
import torch.nn.functional as F

from daisy.model.AbstractRecommender import GeneralRecommender
from daisy.utils.config import initializer_param_config, initializer_config

class NGCF(GeneralRecommender):
    def __init__(self, config):
        """
        Pair-wise NGCF Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        node_dropout: float, node dropout ratio
        mess_dropout : float, messsage dropout rate
        lr : float, learning rate
        reg_2 : float, second-order regularization term
        epochs : int, number of training epochs
        node_dropout_flag: int, NGCF: 0: Disable node dropout, 1: Activate node dropout
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(NGCF, self).__init__(config)

        self.n_user = config['user_num']
        self.n_item = config['item_num']
        self.emb_size = config['factors']

        self.node_dropout = config['node_dropout']
        self.mess_dropout = [config['mess_dropout'], config['mess_dropout'], config['mess_dropout']]
        self.layers = [config['factors'], config['factors'], config['factors']]

        self.norm_adj = config['norm_adj']
        
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.node_dropout_flag = config['node_dropout_flag']

        self.loss_type = config['loss_type']
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'xavier_normal'
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.early_stop = config['early_stop']

        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        self.sparse_norm_adj.to(self.device)

    def init_weight(self):
        initializer = initializer_config[self.initializer]

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update(
                {'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update(
                {'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update(
                {'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape)
        if torch.cuda.is_available():
            random_tensor=random_tensor.cuda()
        else:
            random_tensor=random_tensor.cpu()
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape)
        if torch.cuda.is_available():
            out=out.cuda()
        else:
            out=out.cpu()
        return out * (1. / (1 - rate))

    def forward(self, user, item):
        drop_flag=self.node_dropout_flag
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[user, :]
        i_g_embeddings = i_g_embeddings[item, :]

        scores = torch.sum(torch.mul(u_g_embeddings, i_g_embeddings), dim=1)

        return scores

    def calc_loss(self, batch):
        len_batch = batch.shape[0]
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)

        emb_u = self.embedding_dict['user_emb'][user]
        emb_pos = self.embedding_dict['item_emb'][pos_item]

        pos_pred = self.forward(user, pos_item)
        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)

            loss += self.reg_1 * torch.norm(emb_pos, p=1) / len_batch
            loss += self.reg_2 * torch.norm(emb_pos, p=2) / len_batch
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            emb_neg = self.embedding_dict['item_emb'][neg_item]

            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (torch.norm(emb_pos, p=1) + torch.norm(emb_neg, p=1)) / len_batch
            loss += self.reg_2 * (torch.norm(emb_pos, p=2) + torch.norm(emb_neg, p=2)) / len_batch
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        loss += self.reg_1 * torch.norm(emb_u, p=1) / len_batch
        loss += self.reg_2 * torch.norm(emb_u, p=2) / len_batch

        return loss

    def predict(self, u, i):
        u_g_embeddings = self.embedding_dict['user_emb'][u]
        pos_i_g_embeddings = self.embedding_dict['item_emb'][i]
        pred = torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

        return pred.cpu()

    def rank(self, test_loader):
        pass

    def full_rank(self, u):
        pass
