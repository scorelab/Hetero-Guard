import numpy as np
from deeprobust.graph import utils
import scipy.sparse as sp
import torch

import pynvml


def print_gpu_mem(cuda_device):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(cuda_device)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total / pow(1024, 2)} MB')
    print(f'free     : {info.free / pow(1024, 2)} MB')
    print(f'used     : {info.used / pow(1024, 2)} MB')


def to_tensor(adj=None, features=None, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if adj is not None:
        print(type(adj))
        if sp.issparse(adj):
            adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        else:
            adj = torch.FloatTensor(adj).to(device)
            
    if features is not None:
        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to(device)
        else:
            features = torch.FloatTensor(np.array(features)).to(device)

    if labels is not None:
        labels = torch.LongTensor(labels).to(device)
        
    return adj, features, labels


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)

        r_inv = rowsum.pow(-1/2).flatten()
        
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def row_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    
    if  mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])        
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def row_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """
    device = adj.device
    
    if sparse:
        adj = utils.to_scipy(adj)
        mx = row_normalize_adj(adj)
        return utils.sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj
        if mx[0][0] == 0:
            mx = mx + torch.eye(mx.shape[0]).to(device)

        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx
