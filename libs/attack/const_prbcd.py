"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
"""
from typing import Tuple

import numpy as np
import torch
import torch_sparse

from torch.nn import functional as F
from torch_geometric import utils as pyg_utils

from tqdm import tqdm

from deeprobust.graph import utils
from libs.attack.base_attack import BaseAttack


def get_edge_type_info(data, node_types):
    matching = node_types[0] == node_types[1]
    src = (data.node_type == node_types[0]).nonzero().t()[0]
    tgt = (data.node_type == node_types[1]).nonzero().t()[0]
    
    info = {
        'matching': matching,
        'range': ((src.min(), src.max()), (tgt.min(), tgt.max())),
        'src': src,
        'tgt': tgt,
        'pos': src.size(0) * tgt.size(0) if not matching else int(src.size(0) * (src.size(0) - 1) / 2),
    }
    
    return info


def edge_type_sample(data, edge_type_info, block_size, constraints=[]):
    constrained = {e: k for e, k in data.edge_map.items() if e not in constraints}
    all_info = {e: edge_type_info[e] for e in constrained.keys()}
    all_possible = sum([k['pos'] for k in all_info.values()])
    
    edge_index = []
    
    for idx, e in constrained.items():
        info = all_info[idx]
        n = info['src'].size(0)
        sample_size = int(info['pos'] * block_size / all_possible)
        lin_idx = torch.randint(info['pos'], (sample_size,), device=info['src'].get_device())

        if info['matching']:
            row_col = linear_to_triu_idx(n, lin_idx)
        else:
            row_col = linear_to_mat_idx(n, lin_idx)
        sub_edge_index = torch.stack((info['range'][0][0] + row_col[0], info['range'][1][0] + row_col[1]))
        edge_index.append(sub_edge_index)
        
    edge_index = torch.cat(edge_index, axis=1)
    block = triu_idx_to_linear(data.x.size(0), edge_index[0], edge_index[1])
    return block


def linear_to_mat_idx(m, lin_idx):
    row_idx = (torch.remainder(lin_idx, m)).long()
    col_idx = (torch.div(lin_idx, m)).long()
    return row_idx, col_idx


def triu_idx_to_linear(n, row_idx, col_idx):
    return - (
        row_idx
        - col_idx
        + torch.div(1 - n * (n - 1), 2, rounding_mode='trunc')
        + torch.div((n - row_idx) * ((n - row_idx) - 1), 2, rounding_mode='trunc')
    )


def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + torch.div(1 - n * (n - 1), 2, rounding_mode='trunc')
        + torch.div((n - row_idx) * ((n - row_idx) - 1), 2, rounding_mode='trunc')
    )
    return torch.stack((row_idx, col_idx))


def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


class ConstPRBCD(BaseAttack):
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack.
    data : 
        original graph data.
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    """

    def __init__(self, model, data, block_size, constraints=[], epochs=500, 
                 fine_tune_epochs: int = 50, lr_factor=200, loss_type='CE'):
        super(ConstPRBCD, self).__init__(model, data, True, False)
        
        if self.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device

        self.n: int = data.x.size(0)
        self.keep_heuristic = "WeightOnly"
        self.loss_type: str = loss_type
        self.modified = None
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.epochs_resampling = epochs - fine_tune_epochs
        self.lr_factor = lr_factor
        self.block_size = block_size
        self.constraints = constraints
        self.eps: float = 1e-7
        self.K: int = 20
            
        if hasattr(self.data, 'edge_weight') and self.data.edge_weight is not None:
            self.edge_weight = self.data.edge_weight
        else:
            self.edge_weight = torch.ones((self.data.edge_index.size(1),), device=self.device)
            
        if hasattr(self.data, 'edge_map'):
            self.edge_type_info = {e: get_edge_type_info(self.data, k) for e, k in data.edge_map.items()}
        else:
            raise NotImplementedError('Constrained attack requires a `edge_map` attribute in data')

        self.block: torch.Tensor = None
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None

        self.n_possible_edges = torch.div(self.n * (self.n - 1), 2, rounding_mode='trunc')

    def attack(self, n_perturbations, loss_callback=None, **kwargs):
        assert self.block_size > n_perturbations, \
            f'The block size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'
        
        self.sample_random_block(n_perturbations)

        self.surrogate.eval()
        for t in tqdm(range(self.epochs)):
            self.perturbed_edge_weight.requires_grad = True

            edge_index, edge_weight = self.get_modified_adj()
            
            output = self.surrogate(self.data.x, edge_index, edge_weight)
            mask = self.data['train_mask']            
            loss = self._loss(output[mask], self.data.y[mask])
            if loss_callback is not None:
                loss_callback(float(loss), t)

            grad = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            with torch.no_grad():
                self.update_edge_weights(n_perturbations, t, grad)
            
                if t <= 0:
                    print(f'loss={loss}, grad_abs_sum={grad.abs().sum()}')

                self.projection(n_perturbations, self.eps)

                if t < self.epochs_resampling - 1:
                    self.resample_random_block(n_perturbations)

        edge_index = self.sample_final_edges(n_perturbations)
                
        self.modified = self.data.clone()
        with torch.no_grad():
            self.modified.edge_index, self.modified.edge_weight = self.get_modified_adj()
            keep_indexes = self.modified.edge_weight == 1
            self.modified.edge_weight = self.modified.edge_weight[keep_indexes]
            self.modified.edge_index = self.modified.edge_index[:, keep_indexes]        
#             self.check_adj_tensor(pyg_utils.to_dense_adj(self.modified.edge_index)[0])
    
    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, n_perturbations, eps):
        if torch.clamp(self.perturbed_edge_weight, 0, 1).sum() > n_perturbations:
            left = (self.perturbed_edge_weight - 1).min()
            right = self.perturbed_edge_weight.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.perturbed_edge_weight.data.copy_(
                torch.clamp(self.perturbed_edge_weight.data - miu, min=eps, max=1 - eps)
            )
        else:
            self.perturbed_edge_weight.data.copy_(
                torch.clamp(self.perturbed_edge_weight.data, min=eps, max=1 - eps)
            )

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.perturbed_edge_weight - x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu

        return miu
    
    def get_edge_index(self, adj, complete=True):
        if complete:
            edge_index = (torch.ones_like(adj) - torch.eye(adj.size(0)).to(adj.get_device())).nonzero().t()
        else:
            edge_index = (adj > 0).nonzero().t()
        row, col = edge_index
        edge_weight = adj[row, col]
        return edge_index, edge_weight

    # Modifications
    def sample_random_block(self, n_perturbations: int = 0):
        for _ in range(self.K):
            self.block = edge_type_sample(self.data, self.edge_type_info, self.block_size, self.constraints)
            self.block = torch.unique(self.block, sorted=True)

            self.modified_edge_index = linear_to_triu_idx(self.n, self.block)

            self.perturbed_edge_weight = torch.full_like(
                self.block, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.block.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
        
    def resample_random_block(self, n_perturbations: int):
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.block = self.block[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.K):
            n_edges_resample = self.block_size - self.modified_edge_index.size(1)
            lin_index = edge_type_sample(self.data, self.edge_type_info, n_edges_resample, self.constraints)

            self.block, unique_idx = torch.unique(
                torch.cat((self.block, lin_index)),
                sorted=True,
                return_inverse=True
            )

            self.modified_edge_index = linear_to_triu_idx(self.n, self.block)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.block, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old

            if self.block.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
        
    @torch.no_grad()
    def sample_final_edges(self, n_perturbations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        for i in range(self.K):
            if best_accuracy == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            
            output = self.surrogate(self.data.x, edge_index, edge_weight)
            mask = self.data['train_mask']
            accuracy = (output.argmax(1)[mask] == self.data.y[mask]).float().mean().item()

            # Save best sample
            if best_accuracy > accuracy:
                best_accuracy = accuracy
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))
    
    def update_edge_weights(self, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(float(lr) * gradient)
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps
        
    def get_modified_adj(self):
        if (not self.perturbed_edge_weight.requires_grad):
            modified_edge_index, modified_edge_weight = pyg_utils.to_undirected(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
                        
            non_zero_edges = modified_edge_weight.nonzero().t()[0]
            modified_edge_index = modified_edge_index[:, non_zero_edges]
            modified_edge_weight = modified_edge_weight[non_zero_edges]
                
            edge_index = torch.cat((self.data.edge_index, modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight, modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
        else:
            from torch.utils import checkpoint

            def fuse_edges_run(perturbed_edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                modified_edge_index, modified_edge_weight = pyg_utils.to_undirected(
                    self.modified_edge_index, perturbed_edge_weight, self.n
                )
                
                edge_index = torch.cat((self.data.edge_index, modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight, modified_edge_weight))

                edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
                return edge_index, edge_weight

            with torch.no_grad():
                edge_index = fuse_edges_run(self.perturbed_edge_weight)[0]
            
            edge_weight = checkpoint.checkpoint(
                lambda *input: fuse_edges_run(*input)[1],
                self.perturbed_edge_weight
            )

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight

