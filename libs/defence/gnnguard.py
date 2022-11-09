import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import scipy
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import numpy as np
from deeprobust.graph.utils import *
from torch_geometric.nn import GINConv, global_add_pool, GATConv, GCNConv, ChebConv, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, jaccard_score
from sklearn.metrics.pairwise import euclidean_distances


def att_coef(fea, edge_index):
    # the weights of self-loop
    edge_index = edge_index.tocoo()
#     fea = fea.todense()
    fea_start, fea_end = fea[edge_index.row], fea[edge_index.col]
    isbinray = np.array_equal(fea, fea.astype(bool)) #check is the fea are binary
    np.seterr(divide='ignore', invalid='ignore')
    if isbinray:
        fea_start, fea_end = fea_start.T, fea_end.T
        sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
    else:
        sim_matrix = euclidean_distances(X=fea, Y=fea)
        sim = sim_matrix[edge_index.row, edge_index.col]
        w = 1 / sim
        w[np.isinf(w)] = 0
        sim = w

    """build a attention matrix"""
    att_dense = np.zeros([fea.shape[0], fea.shape[0]], dtype=np.float32)
    row, col = edge_index.row, edge_index.col
    att_dense[row, col] = sim
    if att_dense[0, 0] == 1:
        att_dense = att_dense - np.diag(np.diag(att_dense))
    # normalization, make the sum of each row is 1
    att_dense_norm = normalize(att_dense, axis=1, norm='l1')
    # np.seterr(divide='ignore', invalid='ignore')
    character = np.vstack((att_dense_norm[row, col], att_dense_norm[col, row]))
    character = character.T


    if att_dense_norm[0, 0] == 0:
        # the weights of self-loop
        degree = (att_dense != 0).sum(1)[:, None]
        lam = np.float32(1 / (degree + 1))  # degree +1 is to add itself
        lam = [x[0] for x in lam]
        self_weight = np.diag(lam)
        att = att_dense_norm + self_weight  # add the self loop
    else:
        att = att_dense_norm
    att = np.exp(att) - 1  # make it exp to enhance the difference among edge weights
    att_dense[att_dense <= 0.1] = 0  # set the att <0.1 as 0, this will decrease the accuracy for clean graph

    att_lil = scipy.sparse.lil_matrix(att)
    return att_lil


class GNNGuard(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, device=None):

        super(GNNGuard, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        # self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        nclass = int(nclass)

        self.gc1 = GCNConv(nfeat, nhid, bias=True,)
        self.gc2 = GCNConv(nhid, nclass, bias=True, )

    def forward(self, x, adj_lil):
        '''
            adj: normalized adjacency matrix
        '''
#         x = x.to_dense()
        adj = adj_lil.coalesce().indices()
        edge_weight = adj_lil.coalesce().values()

        x = F.relu(self.gc1(x, adj, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=101, att_0=None,
            attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=500, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None
        self.attention = attention
        if self.attention:
            att_0 = att_coef(features, adj)
            adj = att_0 # update adj
            self.sim = att_0 # update att_0

        self.idx_test = idx_test

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        normalize = False # we don't need normalize here, the norm is conducted in the GCN (self.gcn1) model
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        """Make the coefficient D^{-1/2}(A+I)D^{-1/2}"""
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            # print('iterations:', i)
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            # pred = output[self.idx_test].max(1)[1]

            acc_test =accuracy(output[self.idx_test], labels[self.idx_test])
            # acc_test = pred.eq(labels[self.idx_test]).sum().item() / self.idx_test.shape[0]



            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 200 == 0:
                print('Epoch {}, training loss: {}, test acc: {}'.format(i, loss_train.item(), acc_test))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))


            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test, model_name=None):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])

        split_pred = output.argmax(dim=-1)[idx_test].detach().cpu().numpy()
        split_y = self.labels[idx_test].detach().cpu().numpy() 
        
        acc = accuracy_score(split_y, split_pred)
        f1_micro = f1_score(split_y, split_pred, average='micro')
        f1_macro = f1_score(split_y, split_pred, average='macro')

        auc_pred = np.exp(output[idx_test].detach().cpu().numpy()) if self.nclass > 2 else split_pred
        auc = roc_auc_score(split_y, auc_pred, multi_class='ovr')
                
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc.item()))
        return {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'auc': auc}

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
