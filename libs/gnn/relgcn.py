from typing import Optional, Union, Tuple
from torch_geometric.typing import OptTensor, Adj

import torch
import numpy as np
from torch import Tensor
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_sparse import SparseTensor
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from copy import deepcopy

def masked_edge_weight(edge_weight, edge_mask):
    if isinstance(edge_weight, Tensor):
        return edge_weight[edge_mask]
    else:
        raise NotImplementedError('Only supports tensors')


class WeightedRGCNConv(RGCNConv):

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_weight: OptTensor = None, edge_type: OptTensor = None):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                tmp_weight = masked_edge_weight(edge_weight, edge_type == i) if edge_weight is not None else None

                h = self.propagate(tmp, x=x_l, edge_weight=tmp_weight, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                tmp_weight = masked_edge_weight(edge_weight, edge_type == i) if edge_weight is not None else None

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], edge_weight=tmp_weight, size=size)
                else:
                    h = self.propagate(tmp, x=x_l, edge_weight=tmp_weight, size=size)
                    out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class RelGCN(torch.nn.Module):
    """ 2 Layer Relational Graph Convolutional Network.

    Parameters
    ----------
    num_features : int
        size of input feature dimension
    num_hidden : int
        number of hidden units
    num_classes : int
        size of output dimension
    num_relations : int
        number of relationship types
    num_bases : int
        number of bases in RGCN
    dropout : float
        dropout rate for RGCN
    lr : float
        learning rate for RGCN
    weight_decay : float
        weight decay for RGCN
    """

    def __init__(self, data, num_layers, num_hidden, num_classes, num_relations, num_bases=0, 
                 dropout=0.5, lr=0.01, weight_decay=5e-4, with_bias=True, verbose=False):
        super(RelGCN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_bases = num_bases
        
        self.data = data.to(device)
        
        if isinstance(self.data.x, list):
            self.data.x = [x.to(device) for x in self.data.x]
            self.feat_lin = torch.nn.ModuleList([
                torch.nn.Linear(x.shape[1], num_hidden, dtype=torch.float) for x in self.data.x
            ])
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            in_dim = self.data.x.size(1) if i == 0 and (not hasattr(self, 'feat_lin')) else self.num_hidden
            self.convs.append(
                WeightedRGCNConv(in_dim, self.num_hidden, num_relations, num_bases, bias=with_bias)
            )
        in_dim = self.data.x.size(1) if num_layers == 1 and (not hasattr(self, 'feat_lin')) else self.num_hidden
        self.out_conv = WeightedRGCNConv(in_dim, num_classes, num_relations, num_bases, bias=with_bias)
        
        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, x, edge_index, edge_weight=None, edge_type=None):
        if isinstance(x, list):
            h = []
            for lin, feature in zip(self.feat_lin, x):
                h.append(lin(feature))
            x = torch.cat(h, 0)
            
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight, edge_type)
            x = F.relu(x)
            x= F.dropout(x, self.dropout, training=self.training)
        x = self.out_conv(x, edge_index, edge_weight, edge_type)
        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        if hasattr(self, "feat_lin"):
            for lin in self.feat_lin:
                lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.out_conv.reset_parameters()

    def fit(self, epochs=200, validate=False, **kwargs):
        self.reset_parameters()

        best_loss_val = np.inf
        best_acc_val = 0
        
        history = []
        for epoch in range(1, epochs + 1):
            x, edge_index, edge_type = self.data.x, self.data.edge_index, self.data.edge_type
            train_mask = self.data.train_mask

            # Train
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x, edge_index, edge_type=edge_type)
            loss = F.nll_loss(out[train_mask], self.data.y[train_mask])
            loss.backward()
            self.optimizer.step()
            
            acc = self.test()['acc']
            history.append({
                'train_acc': acc['train'], 
                'val_acc': acc['val' if 'val' in acc else 'test'], 
                'test_acc': acc['test']
            })
            if self.verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {acc["train"]:.4f}, '
                      f'Val: {acc["val" if "val" in acc else "test"]:.4f}, Test: {acc["test"]:.4f}')

            # Validation
            if hasattr(self.data, 'val_mask') and validate:
                val_mask = self.data.val_mask

                self.eval()
                output = self.forward(x, edge_index, edge_type=edge_type)
                loss_val = F.nll_loss(out[val_mask], self.data.y[val_mask])
                acc_val = (output.argmax(dim=-1)[val_mask] == self.data.y[val_mask]).sum() / val_mask.sum()

                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if hasattr(self.data, 'val_mask') and validate:
            if self.verbose:
                print('=== picking the best model according to the performance on validation ===')
            self.load_state_dict(weights)
        return history
    
    @torch.no_grad()
    def test(self, data=None):
        if data is None:
            data = self.data
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        self.eval()
        pred = self.forward(x, edge_index, edge_type=edge_type)
        stat = {'acc': {}, 'f1_micro': {}, 'f1_macro': {}, 'auc': {}}
        for split in ['train', 'val', 'test']:
            mask_name = f'{split}_mask'
            if hasattr(self.data, mask_name):
                mask = self.data[mask_name]
                
                split_pred = pred.argmax(dim=-1)[mask].detach().cpu().numpy()
                split_y = self.data.y[mask].detach().cpu().numpy() 
                acc = accuracy_score(split_y, split_pred)
                f1_mic = f1_score(split_y, split_pred, average='micro')
                f1_mac = f1_score(split_y, split_pred, average='macro')

                auc_pred = np.exp(pred[mask].detach().cpu().numpy()) if self.num_classes > 2 else split_pred
                auc = roc_auc_score(split_y, auc_pred, multi_class='ovr')
                
                stat['acc'][split] = float(acc)
                stat['f1_micro'][split] = float(f1_mic)
                stat['f1_macro'][split] = float(f1_mac)
                stat['auc'][split] = float(auc)
        return stat