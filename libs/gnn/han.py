import numpy as np

import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax

from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def group(xs: List[Tensor], q: nn.Parameter,
          k_lin: nn.Module) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out


class HANConv(MessagePassing):
    r"""
    The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            heads: int = 1,
            negative_slope=0.2,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.Tensor(1, out_channels))

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def forward(
            self, x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj],
            edge_weight_dict: OptTensor,
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            edge_weights = edge_weight_dict[edge_type] if edge_weight_dict is not None else None
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            # propagate_type: (x_dst: PairTensor, alpha: PairTensor)
            out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weights,
                                 alpha=(alpha_src, alpha_dst), size=None)

            out = F.relu(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.q, self.k_lin)

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = out

        return out_dict

    def message(self, x_j: Tensor, edge_weight: OptTensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)

        #         if edge_weight is not None:
        #             print(edge_weight.view(-1, 1).size(), x_j.size(), out.size())
        out = out.view(-1, self.out_channels)
        out = out if edge_weight is None else edge_weight.view(-1, 1) * out
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')


class HAN(torch.nn.Module):
    def __init__(self, data, num_hidden, num_classes, num_heads, head_node, metapaths,
                 dropout=0.5, lr=0.01, weight_decay=0.001, fill_edge_weights=True, verbose=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.num_classes = num_classes
        self.num_hidden = num_hidden

        if metapaths is not None:
            metapath_transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                                                drop_unconnected_nodes=True)
            metapath_neighbours = metapath_transform(data)
            self.data = metapath_neighbours
        else:
            self.data = data

        if fill_edge_weights:
            for edge_store in self.data.edge_stores:
                if 'edge_weight' not in edge_store:
                    edge_weight = torch.ones((edge_store.edge_index.size(1),))
                    edge_store.edge_weight = edge_weight

        self.data = self.data.to(device)

        self.han_conv = HANConv(-1, num_hidden, heads=num_heads,
                                dropout=dropout, metadata=self.data.metadata())
        self.lin = torch.nn.Linear(num_hidden, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.head_node = head_node
        self.to(device)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        out = self.han_conv(x_dict, edge_index_dict, edge_weight_dict)
        out = self.lin(out[self.head_node])
        return F.log_softmax(out, dim=1)

    def fit(self, epochs=100, **kwargs):
        with torch.no_grad():  # Initialize lazy modules.
            pred = self.forward(self.data.x_dict, self.data.edge_index_dict)

        best_loss_val = np.inf
        best_acc_val = 0

        history = []
        for epoch in range(1, epochs + 1):
            train_mask = self.data[self.head_node].train_mask

            # Train
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(self.data.x_dict, self.data.edge_index_dict)
            loss = F.cross_entropy(out[train_mask], self.data[self.head_node].y[train_mask])
            loss.backward()
            self.optimizer.step()

            acc = self.test()['acc']
            history.append({'val_acc': acc['val' if 'val' in acc else 'test'], 'test_acc': acc['test'],
                            'train_acc': acc['train']})
            if self.verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {acc["train"]:.4f}, '
                      f'Val: {acc["val" if "val" in acc else "test"]:.4f}, Test: {acc["test"]:.4f}')

            # Validation
            if hasattr(self.data[self.head_node], 'val_mask'):
                val_mask = self.data[self.head_node].val_mask

                self.eval()
                output = self.forward(self.data.x_dict, self.data.edge_index_dict)
                loss_val = F.cross_entropy(out[val_mask], self.data[self.head_node].y[val_mask])
                acc_val = (output.argmax(dim=-1)[val_mask] == self.data[self.head_node].y[
                    val_mask]).sum() / val_mask.sum()

                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
        if hasattr(self.data[self.head_node], 'val_mask'):
            if self.verbose:
                print('=== picking the best model according to the performance on validation ===')
            self.load_state_dict(weights)
        return history

    @torch.no_grad()
    def test(self):
        x_dict, edge_index_dict = self.data.x_dict, self.data.edge_index_dict

        self.eval()
        pred = self.forward(x_dict, edge_index_dict)
        stat = {'acc': {}, 'f1_micro': {}, 'f1_macro': {}, 'auc': {}}
        for split in ['train', 'val', 'test']:
            mask_name = f'{split}_mask'
            if hasattr(self.data[self.head_node], mask_name):
                mask = self.data[self.head_node][mask_name]

                split_pred = pred.argmax(dim=-1)[mask].detach().cpu().numpy()
                split_y = self.data[self.head_node].y[mask].detach().cpu().numpy()
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
