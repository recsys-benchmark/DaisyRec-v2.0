'''
@author: 
Tinglin Huang (huangtinglin@outlook.com)
https://github.com/huangtinglin/NGCF-PyTorch
Cong Geng
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm


class NGCF(nn.Module):
    def __init__(self, 
                n_user, 
                n_item, 
                norm_adj, 
                factors,
                batch_size,
                node_dropout,
                mess_dropout,
                lr,
                reg_2,
                epochs,
                node_dropout_flag,
                loss_type='BPR',
                early_stop=True,
                gpuid='0'):
        super(NGCF, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True
        
        self.n_user = n_user
        self.n_item = n_item
        self.emb_size = factors
        self.batch_size = batch_size
        self.node_dropout = node_dropout
        self.mess_dropout = [mess_dropout, mess_dropout, mess_dropout]
        self.layers = [factors, factors, factors]
        self.norm_adj = norm_adj
        
        self.reg_2 = reg_2
        self.epochs = epochs
        self.lr = lr
        self.node_dropout_flag=node_dropout_flag

        self.loss_type = loss_type
        self.early_stop=early_stop

        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        if torch.cuda.is_available():
            self.sparse_norm_adj=self.sparse_norm_adj.cuda()
        else:
            self.sparse_norm_adj=self.sparse_norm_adj.cpu()
    
    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
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
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.reg_2 * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
    
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    
    def forward(self, user, item_i, item_j):
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
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

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

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[user, :]
        pos_i_g_embeddings = i_g_embeddings[item_i, :]
        neg_i_g_embeddings = i_g_embeddings[item_j, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
    
    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        #optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item_i, item_j, label in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                    label = label.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()
                    label = label.cpu()

                self.zero_grad()
                emd_u, emd_i, emd_j = self.forward(user, item_i, item_j)

                if self.loss_type == 'BPR':
                    loss, mf_loss, emb_loss= self.create_bpr_loss(emd_u, emd_i, emd_j)
                elif self.loss_type == 'HL':
                    loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
                else:
                    raise ValueError(f'Invalid loss type: {self.loss_type}')

                # loss += self.lamda * (self.embed_item.weight.norm() + self.embed_user.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self, u, i):
        emd_u, emd_i, _ = self.forward(u, i, i)
        pred_i = torch.sum(torch.mul(emd_u, emd_i), dim=1)
        return pred_i.cpu()