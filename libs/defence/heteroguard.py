from typing import Union, Tuple, Dict, Optional
from torch_geometric.typing import OptTensor, OptPairTensor, NodeType, EdgeType, Adj, Size

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict
# from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.hgt_conv import group
from collections import defaultdict
from torch_scatter import scatter_add

import numpy as np
import warnings


class NeighbourMeanConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class HeteroGuardConv(Module):
    def __init__(self, convs: Dict[EdgeType, Module]):
        super().__init__()

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behaviour.")

        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *args_dict,
    ) -> Dict[NodeType, Tensor]:
        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            conv = self.convs[str_edge_type]

            if src == dst:
                out = conv(x_dict[src], edge_index, *args)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args)

            out_dict[edge_type] = out
        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
    

class HeteroGuardAttn(torch.nn.Module):
    def __init__(self, data, normalize=True, exp=True, drop_bottom=True, gated=True, verbose=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.normalize = normalize
        self.exp = exp
        self.drop_bottom = drop_bottom
        self.gated = gated
        self.verbose = verbose
        
        edge_types = data.metadata()[1]
        convs = {
            et: NeighbourMeanConv((-1, -1), data[et[0]].x.size(1)) for et in edge_types if not (et[0] == et[-1])
        }
        self.conv = HeteroGuardConv(convs)
        self.to(device)
        
    def post_attn(self, attn, edge_index, num_nodes):
        # Row normalize
        if self.normalize:
            deg = scatter_add(attn, edge_index[0], dim=0, dim_size=num_nodes)
            deg_inv = deg.pow(-1.0)
            attn = deg_inv[edge_index[0]] * attn
            attn = torch.nan_to_num(attn)

        if self.exp:
            attn = (torch.exp(attn) - 1) / (np.exp(1) - 1)  # make it exp to enhance the difference among edge weights
            
        if self.drop_bottom:
            attn[attn <= 0.1] = 0  # set the att <0.1 as 0, this will decrease the accuracy for clean graph
        return attn
        
    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        hop_dict = self.conv(x_dict, edge_index_dict, edge_weight_dict)
        
        attn_dict = {}
        for (src, rel, dst), edges in edge_index_dict.items():
            src_feat = x_dict[src]
            dst_feat = src_feat if src == dst else hop_dict[(src, rel, dst)]
            
            src_feat = src_feat[edges[0]]
            dst_feat = dst_feat[edges[1]]
            
            attn = F.cosine_similarity(src_feat, dst_feat)
            attn = self.post_attn(attn, edges, x_dict[src].size(0))
                
            attn_dict[(src, rel, dst)] = attn.to(self.device)

        return attn_dict