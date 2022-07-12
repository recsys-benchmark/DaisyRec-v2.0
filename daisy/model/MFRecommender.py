import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class MF(GeneralRecommender):
    def __init__(self, config):
        """
        Matrix Factorization Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(MF, self).__init__(config)
        
        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']

        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'sgd'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        pred = (embed_user * embed_item).sum(dim=-1)

        return pred

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).weight.norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).weight.norm(p=1) + self.embed_item(neg_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).weight.norm() + self.embed_item(neg_item).weight.norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        # add regularization term
        loss += self.reg_1 * (self.embed_user(user).weight.norm(p=1))
        loss += self.reg_2 * (self.embed_user(user).weight.norm())

        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu()
        
        return pred

    def rank(self, test_loader):
        pass
