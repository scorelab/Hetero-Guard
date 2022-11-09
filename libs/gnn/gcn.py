import torch
import numpy as np
from torch.nn import functional as F
from collections import defaultdict

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, fill_diag, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from copy import deepcopy

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    num_hidden : int
        number of hidden units
    num_classes : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay for GCN
    """

    def __init__(self, data, num_layers, num_hidden, num_classes, dropout=0.5, lr=0.01, weight_decay=0.001,
                 verbose=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.data = data.to(device)

        if isinstance(self.data.x, list):
            self.data.x = [x.to(device) for x in self.data.x]
            self.feat_lin = torch.nn.ModuleList([
                torch.nn.Linear(x.shape[1], num_hidden, dtype=torch.float) for x in self.data.x
            ])

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(-1, num_hidden))
        self.out_conv = GCNConv(-1, num_classes)

        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, list):
            h = []
            for lin, feature in zip(self.feat_lin, x):
                h.append(lin(feature))
            x = torch.cat(h, 0)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_conv(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        if hasattr(self, "feat_lin"):
            for lin in self.feat_lin:
                lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.out_conv.reset_parameters()

    def fit(self, epochs=200, patience=20, patience_delta=0.005, validate=False, loss_callback=None, **kwargs):
        with torch.no_grad():  # Initialize lazy modules.
            self.forward(self.data.x, self.data.edge_index)

        self.reset_parameters()
        best_loss_val = np.inf
        best_acc_val = 0

        for epoch in range(1, epochs + 1):
            x, edge_index = self.data.x, self.data.edge_index
            train_mask = self.data.train_mask

            # Train
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x, edge_index)
            loss = F.nll_loss(out[train_mask], self.data.y[train_mask])
            loss.backward()
            self.optimizer.step()

            acc = self.test()

            if loss_callback is not None:
                loss_callback(float(loss), epoch)
            if self.verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {acc["train"]["acc"]:.4f}, '
                      f'Val: {acc["val" if "val" in acc else "test"]["acc"]:.4f}, Test: {acc["test"]["acc"]:.4f}')

            # Validation
            if hasattr(self.data, 'val_mask') and validate:
                val_mask = self.data.val_mask

                self.eval()
                output = self.forward(x, edge_index)
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

    @torch.no_grad()
    def test(self):
        x, edge_index = self.data.x, self.data.edge_index

        self.eval()
        pred = self.forward(x, edge_index)
        stat = defaultdict(dict)
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

                stat[split]['acc'] = float(acc)
                stat[split]['f1_micro'] = float(f1_mic)
                stat[split]['f1_macro'] = float(f1_mac)
                stat[split]['auc'] = float(auc)
        return stat
