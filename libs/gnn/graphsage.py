from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from collections import defaultdict

import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class GraphSAGE(torch.nn.Module):
    def __init__(self, data, num_layers, num_hidden, num_classes, 
                 dropout=0.5, lr=0.01, weight_decay=0.001, verbose=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.data = data.to(device)
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv((-1, -1), num_hidden))
        self.out_conv = SAGEConv((-1, -1), num_classes)

        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x= F.dropout(x, self.dropout, training=self.training)
        x = self.out_conv(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_conv.reset_parameters()

    def fit(self, train_loader, epochs=200, patience=20, patience_delta=0.005, validate=False, loss_callback=None, **kwargs):
        with torch.no_grad():  # Initialize lazy modules.
            pred = self.forward(self.data.x, self.data.edge_index)

        self.reset_parameters()

        best_loss_val = np.inf
        best_acc_val = 0
        
        for epoch in range(1, epochs + 1):
            
            total_loss = total_examples = 0
            for data in train_loader:
                data = data.to(self.device)
                batch_size = data.batch_size
                
                x, edge_index = data.x, data.edge_index
                train_mask = data.train_mask[:batch_size]

                self.optimizer.zero_grad()
                out = self.forward(x, edge_index)
                loss =  F.nll_loss(out[:batch_size][train_mask], data.y[:batch_size][train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * batch_size
                total_examples += batch_size
                        
            if loss_callback is not None:
                loss_callback(float(total_loss), epoch)
                
            if self.verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

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
    def test(self, test_loader):        
        self.eval()

        out, mask, y = [], [], []
        for data in test_loader:
            data = data.to(self.device)
            batch_size = data.batch_size
            
            out.append(self.forward(data.x, data.edge_index)[:batch_size])
            mask.append(data.test_mask[:batch_size])
            y.append(data.y[:batch_size])
            
        pred = torch.cat(out, dim=0).cpu()
        mask = torch.cat(mask, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()
        
        stat = defaultdict(dict)

        split_pred = pred.argmax(dim=-1)[mask].detach().cpu().numpy()
        split_y = y[mask].detach().cpu().numpy() 
        acc = accuracy_score(split_y, split_pred)
        f1_mic = f1_score(split_y, split_pred, average='micro')
        f1_mac = f1_score(split_y, split_pred, average='macro')

        auc_pred = np.exp(pred[mask].detach().cpu().numpy()) if self.num_classes > 2 else split_pred
        auc = roc_auc_score(split_y, auc_pred, multi_class='ovr')

        stat['test']['acc'] = float(acc)
        stat['test']['f1_micro'] = float(f1_mic)
        stat['test']['f1_macro'] = float(f1_mac)
        stat['test']['auc'] = float(auc)
        return stat


class HeteroSAGE(torch.nn.Module):
    def __init__(self, data, num_layers, num_hidden, num_classes, head_node, attension=None, gated_attension=False,
                 dropout=0.5, lr=0.01, weight_decay=0.001, fill_edge_weights=True, verbose=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.attension = attension is not None
        self.gated = gated_attension
        
        if fill_edge_weights:
            for edge_store  in data.edge_stores:
                if 'edge_weight' not in edge_store:
                    edge_weight = torch.ones((edge_store.edge_index.size(1),))
                    edge_store.edge_weight = edge_weight
        self.data = data.to(device)
        
        if self.attension:
            self.hguard = attension
        else:
            assert not gated_attension , "gated_attension cannot be set to True without attension"
            
        if self.gated:
            self.gate = torch.nn.Parameter(torch.rand(1))
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(HeteroConv({
                edge_type: SAGEConv((-1, -1), num_hidden) for edge_type in self.data.metadata()[1]
            }, aggr='sum'))
        self.out_conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), num_classes) for edge_type in self.data.metadata()[1]
        }, aggr='sum')
        
        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.head_node = head_node
        self.to(device)
        
#         print(list(self.parameters()))
        
    def plot_attn(self, attn):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for edge_type, all_weights in attn.items():
            plt.figure(figsize=(12, 6))
            sns.histplot(all_weights.cpu(), stat='percent', bins=100)
            plt.title(f"Histogram of edge weights: {edge_type}")
            plt.xlabel('Edge Weight')
        
    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        prev_attn = None
        for i in range(len(self.convs)):
            if self.attension:
                with torch.no_grad():
                    attn = self.hguard(x_dict, edge_index_dict, edge_weight_dict)
                if self.gated:
                    attn_value = {
                        et: self.gate * prev_attn[et] + (1 - self.gate) * et_attn for et, et_attn in attn.items()
                    } if i > 0 else attn
                    prev_attn = attn
                else:
                    attn_value = attn
                x_dict = self.convs[i](x_dict, edge_index_dict, attn_value)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict, edge_weight_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, self.dropout, training=self.training) for key, x in x_dict.items()}
            
        if self.attension:
            with torch.no_grad():
                attn = self.hguard(x_dict, edge_index_dict, edge_weight_dict)
            attn_value = {
                et: self.gate * prev_attn[et] + (1 - self.gate) * et_attn for et, et_attn in attn.items()
            } if prev_attn is not None else attn
            x = self.out_conv(x_dict, edge_index_dict, attn_value)[self.head_node]
        else:
            x = self.out_conv(x_dict, edge_index_dict, edge_weight_dict)[self.head_node]
        
        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_conv.reset_parameters()

    def fit(self, epochs=200, validate=False, **kwargs):
        with torch.no_grad():  # Initialize lazy modules.
            pred = self.forward(self.data.x_dict, self.data.edge_index_dict, self.data.edge_weight_dict)

        self.reset_parameters()

        best_loss_val = np.inf
        best_acc_val = 0
        
        history = []
        for epoch in range(1, epochs + 1):
            x_dict, edge_index_dict, edge_weight_dict = self.data.x_dict, self.data.edge_index_dict, self.data.edge_weight_dict
            train_mask = self.data[self.head_node].train_mask

            # Train
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x_dict, edge_index_dict, edge_weight_dict)
            loss = F.nll_loss(out[train_mask], self.data[self.head_node].y[train_mask])
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
            if hasattr(self.data[self.head_node], 'val_mask') and validate:
                val_mask = self.data[self.head_node].val_mask

                self.eval()
                output = self.forward(x_dict, edge_index_dict, edge_weight_dict)
                loss_val = F.nll_loss(out[val_mask], self.data[self.head_node].y[val_mask])
                acc_val = (output.argmax(dim=-1)[val_mask] == self.data[self.head_node].y[val_mask]).sum() / val_mask.sum()

                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
#             history.append({'val_acc': float(best_acc_val), 'test_acc': acc['test']})

        if hasattr(self.data[self.head_node], 'val_mask') and validate:
            if self.verbose:
                print('=== picking the best model according to the performance on validation ===')
            self.load_state_dict(weights)
        return history

    @torch.no_grad()
    def test(self):        
        x_dict, edge_index_dict, edge_weight_dict = self.data.x_dict, self.data.edge_index_dict, self.data.edge_weight_dict
        
        self.eval()
        pred = self.forward(x_dict, edge_index_dict, edge_weight_dict)#.argmax(dim=-1)
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