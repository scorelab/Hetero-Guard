import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph import utils


class BaseAttack(torch.nn.Module):
    """Abstract base class for target attack classes.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    """

    def __init__(self, model, data, attack_structure=True, attack_features=False):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.data = data
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.modified = None
        
        if model is not None:
            self.device = model.device
            self.num_classes = model.num_classes
            self.num_hidden = model.num_hidden

    def attack(self, n_perturbations, **kwargs):
        """Generate attacks on the input graph.

        Parameters
        ----------
        n_perturbations : int
            Number of edge removals/additions.

        Returns
        -------
        None.

        """
        pass

#     def check_adj(self, adj):
#         """Check if the modified adjacency is symmetric and unweighted.
#         """
#         assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
#         assert adj.tocsr().max() == 1, "Max value should be 1!"
#         assert adj.tocsr().min() == 0, "Min value should be 0!"

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"