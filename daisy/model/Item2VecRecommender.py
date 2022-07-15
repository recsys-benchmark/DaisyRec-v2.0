import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class Item2Vec(GeneralRecommender):
    def __init__(self, config):
        '''
        Item2Vec Recommender class

        Parameters
        ----------
        factors : int
            item embedding dimension
        lr : float
            learning rate
        epochs : int
            No. of training iterations
        '''
        super(Item2Vec, self).__init__(config)

        self.user_embedding = nn.Embedding(config['user_num'], config['factors'])
        self.ur = config['train_ur']

        self.shared_embedding = nn.Embedding(config['item_num'], config['factors'])
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.out_act = nn.Sigmoid()

        # default loss function for item2vec is cross-entropy
        self.loss_type = 'CL'

        self.apply(self._init_weight)

    def forward(self, target_i, context_j):
        target_emb = self.shared_embedding(target_i) # batch_size * embedding_size
        context_emb = self.shared_embedding(context_j) # batch_size * embedding_size
        output = torch.sum(target_emb * context_emb, dim=1)
        output = self.out_act(output)

        return output.view(-1)

    def fit(self, train_loader):
        super().fit(train_loader)

        print('Start building user embedding...')
        for u in self.ur.keys():
            uis = torch.tensor(list(self.ur[u]), device=self.device)
            self.user_embedding.weight.data[u] = self.shared_embedding(uis).sum(dim=0)

    def calc_loss(self, batch):
        target_i = batch[0].to(self.device)
        context_j = batch[1].to(self.device)
        label = batch[2].to(self.device)
        prediction = self.forward(target_i, context_j)
        loss = self.criterion(prediction, label)
        
        return loss
    
    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        
        user_emb = self.user_embedding(u)
        item_emb = self.shared_embedding(i)
        pred = (user_emb * item_emb).sum(dim=-1)
        
        return pred.cpu()

    def rank(self, test_loader):
        pass

    def full_rank(self, u):
        pass

