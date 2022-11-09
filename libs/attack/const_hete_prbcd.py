from typing import Tuple

import numpy as np
import torch
import torch_sparse

from torch.nn import functional as F
from torch_geometric import utils as pyg_utils

from tqdm import tqdm

from deeprobust.graph import utils
from libs.attack.base_attack import BaseAttack


def mat_idx_to_linear(m, row_idx, col_idx):
    return row_idx * m + col_idx


def linear_to_mat_idx(m, lin_idx):
    col_idx = torch.remainder(lin_idx, m).long()
    row_idx = torch.div(lin_idx, m, rounding_mode='trunc').long()
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


def check_sanity(hete_sym, edge_index_dict, edge_weight_dict):
    for left, right in hete_sym.items():
        index = torch.eq(edge_index_dict[left], edge_index_dict[right].flip(dims=[0]))    
        weight = torch.eq(edge_weight_dict[left], edge_weight_dict[right])
        assert index.unique()[0] == True and weight.unique()[0] == True, "Sanity check failed"


class ConstHetePRBCD(BaseAttack):
    """Heterogeneous PR-BCD attack for graph data.

    Parameters
    ----------
    model :
        model to attack.
    data : 
        original `HeteroData` graph data.
    block_size:
        sample size for the randomized block descent
    head_node:
        head_node of the heterogeneous graph
    budget: 
        edge type to attack with the perturbation rate
    epochs:
        number of total epochs to run the attack
    fine_tune_epochs:
        number of fine tune epochs (optimize without resampling). should be less than `epochs`
    lr_factor:
        learning rate factor
    lamb:
        lambda value (rate of existing edge sampling)
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    """

    def __init__(self, model, data, block_size, head_node, budget, hete_symmetric={}, epochs=200, 
                 fine_tune_epochs: int = 50, lr_factor=200, lamb=None, loss_type='CE'):
        super(ConstHetePRBCD, self).__init__(model, data, True, False)
        
        if lamb is not None:
            assert len(budget) > 0, "budget should contain at least one edge_type"
            assert fine_tune_epochs < epochs, "`fine_tune_epochs should be less than `epochs`"
            assert isinstance(lamb, float) and lamb >= 0 and lamb <= 1, "mie should be either `none` or float between 0 and 1"
            
        if self.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device

        self.n: int = data.num_nodes
        self.keep_heuristic = "WeightOnly"
        
        self.block_size = block_size
        self.head_node = head_node
        self.budget = [et for et in budget if et in data.edge_types]
        self.hete_sym = hete_symmetric
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.lr_factor = lr_factor
        self.lamb = lamb
        self.loss_type: str = loss_type
            
        self.modified = None
        self.epochs_resampling = epochs - fine_tune_epochs
        self.eps: float = 1e-7
        self.K: int = 20
            
        if hasattr(self.data, 'edge_weight_dict'):
            self.edge_weight_dict = self.data.edge_weight_dict
        else:
            raise NotImplementedError("Hetero Attack requires edge_weight to be avalible in data!")

        self.edge_type_info = ConstHetePRBCD.get_edge_type_info(self.data)
        self.constrained_edge_index = {et: ei for et, ei in data.edge_index_dict.items() if et in self.budget}
        self.constrained_edge_weight = {et: ew for et, ew in data.edge_weight_dict.items() if et in self.budget}
        
        self.et_to_i = {et: i for i, et in enumerate(data.edge_types)}
        self.i_to_et = {i: et for et, i in self.et_to_i.items()}

        self.block = {}
        self.modified_edge_index = {}
        self.perturbed_edge_weight = {}

    def attack(self, n_perturbations, check_modified=True, **kwargs):
        assert self.block_size > n_perturbations, \
            f'The block size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'
        
        self.sample_random_block(n_perturbations)

        self.surrogate.eval()
        history = []
        for t in tqdm(range(self.epochs)):
            for et in self.perturbed_edge_weight:
                self.perturbed_edge_weight[et].requires_grad = True
                
            edge_index_dict, edge_weight_dict = self.get_modified_adj()
            output = self.surrogate(self.data.x_dict, edge_index_dict, edge_weight_dict)
        
            mask = self.data[self.head_node]['train_mask']
            loss = self._loss(output[mask], self.data[self.head_node].y[mask])
            history.append(loss.cpu().detach().numpy())
            
            input_list = list(self.perturbed_edge_weight.items())
            grad = torch.autograd.grad(loss, [pew for et, pew in input_list])

            with torch.no_grad():
                for i, (edge_type, _) in enumerate(input_list):
                    self.update_edge_weights(edge_type, n_perturbations, t, grad[i])
            
#                     if t <= 0:
#                         print(f'({"_".join(edge_type)}): loss={loss}, grad_abs_sum={grad[i].abs().sum()}')

                    self.projection(n_perturbations, self.eps)

                if t < self.epochs_resampling - 1:
                    self.resample_random_block(n_perturbations)

        edge_index = self.sample_final_edges(n_perturbations)
                
        self.modified = self.data.clone()
        with torch.no_grad():
            edge_index_dict, edge_weight_dict = self.get_modified_adj()

            for edge_type in edge_weight_dict:
                keep_indexes = edge_weight_dict[edge_type] == 1
                edge_weight_dict[edge_type] = edge_weight_dict[edge_type][keep_indexes]
                edge_index_dict[edge_type] = edge_index_dict[edge_type][:, keep_indexes]
                
                del self.modified[edge_type]
                self.modified[edge_type].edge_index = edge_index_dict[edge_type]
                self.modified[edge_type].edge_weight = edge_weight_dict[edge_type]
            
        if check_modified:
            check_sanity(self.hete_sym, self.modified.edge_index_dict, self.modified.edge_weight_dict)
        return np.array(history)
    
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
    
    def dict_to_linear(self, perturbed_edge_weight, return_index=False):
        edge_weights, edge_types, edge_index = [], [], []

        for edge_type, edge_weight in perturbed_edge_weight.items():
            edge_weights.append(edge_weight)
            edge_types.append(torch.full_like(edge_weight, self.et_to_i[edge_type], dtype=torch.long))
            if return_index:
                edge_index.append(edge_weight.nonzero().t()[0])
        if return_index:
            return torch.cat(edge_weights), torch.cat(edge_types), torch.cat(edge_index)
        return torch.cat(edge_weights), torch.cat(edge_types)
    
    def linear_to_dict(self, edge_weights, edge_types):
        perturbed_edge_weight = {}
        for i in edge_types.unique():
            type_edge_weights_i = edge_types == i
            perturbed_edge_weight[self.i_to_et[int(i)]] = edge_weights[type_edge_weights_i]
        return perturbed_edge_weight

    def projection(self, n_perturbations, eps):
        edge_weights, edge_types = self.dict_to_linear(self.perturbed_edge_weight)
        
        if torch.clamp(edge_weights, 0, 1).sum() > n_perturbations:
            left = (edge_weights - 1).min()
            right = edge_weights.max()
            miu = self.bisection(left, right, n_perturbations, edge_weights, epsilon=1e-5)
            edge_weights.data.copy_(
                torch.clamp(edge_weights.data - miu, min=eps, max=1 - eps)
            )
        else:
            edge_weights.data.copy_(
                torch.clamp(edge_weights.data, min=eps, max=1 - eps)
            )
        
        for edge_type, edge_weight in self.linear_to_dict(edge_weights, edge_types).items():
            self.perturbed_edge_weight[edge_type] = edge_weight

    def bisection(self, a, b, n_perturbations, edge_weights, epsilon):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

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
    
    @staticmethod
    def get_edge_type_info(data):
        info = {}
        for (src, et, tgt) in data.edge_types:
            matching = src == tgt
            src_size, tgt_size = data[src].num_nodes, data[tgt].num_nodes

            info[(src, et, tgt)] = {
                'matching': matching,
                'size': (src_size, tgt_size),
                'pos': src_size * tgt_size if not matching else int(src_size * (src_size - 1) / 2),
            }

        return info

    def edge_type_sample(self, data, block_size, resample=False):
        all_possible = sum([k['pos'] for e, k in self.edge_type_info.items() if e in self.budget])

        block = {}
        block_edge_index = {}

        for (src, et, tgt) in self.budget:
            edge_index = []

            info = self.edge_type_info[(src, et, tgt)]
            n, m = self.data[src].num_nodes, self.data[tgt].num_nodes
            sample_size = int(info['pos'] * block_size / all_possible)
                        
            # Sample from the existing
            if self.lamb is not None:
                existing_sample_size = int(sample_size * self.lamb)
                existing_edges = self.data[(src, et, tgt)].edge_index
                
                population = existing_edges.size(1)
                if existing_sample_size > population:
                    existing_sample_size = population
                
                if existing_sample_size > 0:
                    existing_lin_idx = torch.randint(population, (existing_sample_size,), device=self.device)
                    edge_index.append(existing_edges[:, existing_lin_idx])
                
                sample_size = sample_size - existing_sample_size

            # Sample from the population
            lin_idx = torch.randint(info['pos'], (sample_size,), device=self.device)

            if info['matching']:
                row_col = linear_to_triu_idx(n, lin_idx)
            else:
                row_col = linear_to_mat_idx(m, lin_idx)
            sub_edge_index = torch.stack((row_col[0], row_col[1]))
            
            if self.lamb is not None:
                if info['matching']:
                    existing_edges_i = triu_idx_to_linear(n, existing_edges[0], existing_edges[1])
                else:
                    existing_edges_i = mat_idx_to_linear(m, existing_edges[0], existing_edges[1])

                def is_exist(a):
                    return a not in existing_edges_i
                
                sub_edge_index_i = [is_exist(a) for a in torch.unbind(lin_idx, dim=0)]
                sub_edge_index = sub_edge_index[:, sub_edge_index_i]

            edge_index.append(sub_edge_index)
            edge_index = torch.cat(edge_index, axis=1)

            block_edge_index[(src, et, tgt)] = edge_index
            if info['matching']:
                block[(src, et, tgt)] = triu_idx_to_linear(n, edge_index[0], edge_index[1])
            else:
                block[(src, et, tgt)] = mat_idx_to_linear(m, edge_index[0], edge_index[1])

        return block, block_edge_index

    def sample_random_block(self, n_perturbations):
        for _ in range(self.K):
            self.block, edge_index = self.edge_type_sample(self.data, self.block_size)

            for edge_type in self.block:
                block = self.block[edge_type]
                self.block[edge_type], unique_idx = torch.unique(block, sorted=True, return_inverse=True)
                
                edge_index[edge_type][:, unique_idx] = edge_index[edge_type].clone()
                self.modified_edge_index[edge_type] = edge_index[edge_type][:, :self.block[edge_type].size(0)]

                self.perturbed_edge_weight[edge_type] = torch.full_like(
                    self.block[edge_type], self.eps, dtype=torch.float32, requires_grad=True
                )

            if sum([block.size(0) for et, block in self.block.items()]) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
        
    def resample_random_block(self, n_perturbations):
        edge_weights, edge_types, edge_index = self.dict_to_linear(self.perturbed_edge_weight, return_index=True)
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(edge_weights)
            idx_keep = (edge_weights <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        edge_types = edge_types[sorted_idx]
        edge_index = edge_index[sorted_idx]

        for edge_type in self.modified_edge_index:
            edge_type_i = edge_types == self.et_to_i[edge_type]
            type_sorted_idx = edge_index[edge_type_i]
            
            self.block[edge_type] = self.block[edge_type][type_sorted_idx]
            self.modified_edge_index[edge_type] = self.modified_edge_index[edge_type][:, type_sorted_idx]
            self.perturbed_edge_weight[edge_type] = self.perturbed_edge_weight[edge_type][type_sorted_idx]
            
        existing_block_size = sum([block.size(0) for block in self.block.values()])
        
        # Sample until enough edges were drawn
        for i in range(self.K):
            n_edges_resample = self.block_size - existing_block_size
            lin_index, edge_index = self.edge_type_sample(self.data, n_edges_resample, True)
            
            for edge_type in lin_index:
                self.block[edge_type], unique_idx = torch.unique(
                    torch.cat((self.block[edge_type], lin_index[edge_type])),
                    sorted=True,
                    return_inverse=True
                )
                
                modified_edge_index = torch.cat([self.modified_edge_index[edge_type], edge_index[edge_type]], axis=1)
                modified_edge_index[:, unique_idx] = modified_edge_index.clone()
                self.modified_edge_index[edge_type] = modified_edge_index[:, :self.block[edge_type].size(0)]
            
                # Merge existing weights with new edge weights
                perturbed_edge_weight_old = self.perturbed_edge_weight[edge_type].clone()
                self.perturbed_edge_weight[edge_type] = torch.full_like(self.block[edge_type], self.eps, dtype=torch.float32)
                self.perturbed_edge_weight[edge_type][
                    unique_idx[:perturbed_edge_weight_old.size(0)]
                ] = perturbed_edge_weight_old

            if sum([block.size(0) for et, block in self.block.items()]) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
        
    @torch.no_grad()
    def sample_final_edges(self, n_perturbations) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        
        perturbed_edge_weight = {}
        for edge_type in self.perturbed_edge_weight:
            perturbed_edge_weight[edge_type] = self.perturbed_edge_weight[edge_type].detach()
            perturbed_edge_weight[edge_type][perturbed_edge_weight[edge_type] <= self.eps] = 0

        for i in range(self.K):
            edge_weights, edge_types = self.dict_to_linear(perturbed_edge_weight)
            if best_accuracy == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(edge_weights)
                sampled_edges[torch.topk(edge_weights, int(n_perturbations)).indices] = 1
            else:
                sampled_edges = torch.bernoulli(edge_weights).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                continue
                
            for edge_type, sampled_edge_weight in self.linear_to_dict(sampled_edges, edge_types).items():
                self.perturbed_edge_weight[edge_type] = sampled_edge_weight

            edge_index_dict, edge_weight_dict = self.get_modified_adj()
            
            output = self.surrogate(self.data.x_dict, edge_index_dict, edge_weight_dict)
            mask = self.data[self.head_node]['train_mask']
            accuracy = (output.argmax(1)[mask] == self.data[self.head_node].y[mask]).float().mean().item()

            # Save best sample
            if best_accuracy > accuracy:
                best_accuracy = accuracy
                best_edges = {et: pew.clone().cpu() for et, pew in self.perturbed_edge_weight.items()}

        # Recover best sample
        for edge_type in self.perturbed_edge_weight:
            self.perturbed_edge_weight[edge_type].data.copy_(best_edges[edge_type].to(self.device))
    
    def update_edge_weights(self, edge_type: tuple, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight[edge_type].data.add_(float(lr) * gradient)
        self.perturbed_edge_weight[edge_type].data[self.perturbed_edge_weight[edge_type] < self.eps] = self.eps
        
    def get_modified_adj(self):
        if (not any(w.requires_grad for w in self.perturbed_edge_weight.values())):  
            edge_index_dict = {et: ei for et, ei in self.constrained_edge_index.items()}
            edge_weight_dict = {et: ew for et, ew in self.constrained_edge_weight.items()}
            for edge_type, modified_edge_index in self.modified_edge_index.items():
                n, m = self.edge_type_info[edge_type]['size']
                modified_edge_weight = self.perturbed_edge_weight[edge_type]
                
                if self.edge_type_info[edge_type]['matching']:
                    modified_edge_index, modified_edge_weight = pyg_utils.to_undirected(
                        modified_edge_index, modified_edge_weight, n
                    )                    

                non_zero_edges = modified_edge_weight.nonzero().t()[0]
                modified_edge_index = modified_edge_index[:, non_zero_edges]
                modified_edge_weight = modified_edge_weight[non_zero_edges]
                
                edge_index = torch.cat((self.data.edge_index_dict[edge_type], modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.data.edge_weight_dict[edge_type], modified_edge_weight))
                
                edge_index, edge_weight = torch_sparse.coalesce(
                    edge_index, edge_weight, m=n, n=m, op='sum'
                )
                edge_index_dict[edge_type], edge_weight_dict[edge_type] = edge_index, edge_weight
                
                # Hetero symetric
                if (not self.edge_type_info[edge_type]['matching']) and edge_type in self.hete_sym:
                    sym = self.hete_sym[edge_type]
                    edge_index_dict[sym], edge_weight_dict[sym] = edge_index.flip(dims=[0]), edge_weight
        else:
            from torch.utils import checkpoint

            def fuse_edges(perturbed_edge_weights) -> Tuple[list, list]:
                edge_index_dict = {et: ei for et, ei in self.constrained_edge_index.items()}
                edge_weight_dict = {et: ew for et, ew in self.constrained_edge_weight.items()}
                for edge_type, modified_edge_index in self.modified_edge_index.items():
                    n, m = self.edge_type_info[edge_type]['size']
                    modified_edge_weight = perturbed_edge_weights[edge_type]

                    if self.edge_type_info[edge_type]['matching']:
                        modified_edge_index, modified_edge_weight = pyg_utils.to_undirected(
                            modified_edge_index, modified_edge_weight, n
                        )

                    edge_index = torch.cat((self.data.edge_index_dict[edge_type], modified_edge_index), dim=-1)
                    edge_weight = torch.cat((self.data.edge_weight_dict[edge_type], modified_edge_weight))

                    edge_index, edge_weight = torch_sparse.coalesce(
                        edge_index, edge_weight, m=n, n=m, op='sum'
                    )
                    edge_index_dict[edge_type], edge_weight_dict[edge_type] = edge_index, edge_weight
                    
                    # Hetero symetric
                    if (not self.edge_type_info[edge_type]['matching']) and edge_type in self.hete_sym:
                        sym = self.hete_sym[edge_type]
                        edge_index_dict[sym], edge_weight_dict[sym] = edge_index.flip(dims=[0]), edge_weight

                return edge_index_dict, edge_weight_dict

            with torch.no_grad():
                edge_index_dict = fuse_edges(self.perturbed_edge_weight)[0]
            
            edge_weight_dict = fuse_edges(self.perturbed_edge_weight)[1]

        # Allow removal of edges
        for edge_type, edge_weight in edge_weight_dict.items():
            edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        
        return edge_index_dict, edge_weight_dict

