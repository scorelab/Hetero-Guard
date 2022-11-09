import numpy as np
import functools
from typing import Optional, Callable

from libs import graph_utils
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import pandas as pd

import torch
from torch_geometric.data import (InMemoryDataset, HeteroData, Data)
from torch_geometric import transforms as T
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.datasets import IMDB, DBLP


class HomoDNS(Data):
    @classmethod
    def from_Data(cls, data: Data):
        obj = cls()
        for key, value in data.__dict__.items():
            obj.__dict__[key] = value
        return obj
    
    def update_edge_index(self, adjs):
        modified_edge_indexes = []
        edge_types = []

        for idx, adj in enumerate(adjs):
            edge_index = torch.LongTensor(np.array(adj.nonzero()))
            modified_edge_indexes.append(edge_index)

            tensor = torch.ones((edge_index.shape[1],), dtype=torch.float64)
            edge_type = tensor.new_full((edge_index.shape[1],), idx)
            edge_types.append(edge_type)

        modified_edge_indexes = torch.cat(modified_edge_indexes, axis=1)
        edge_types = torch.cat(edge_types)

        self.edge_index = modified_edge_indexes
        self.edge_type = edge_types

        import numpy as np


class DNS(InMemoryDataset):
    def __init__(self, root: str,  num_test=0.3, num_val=0.2,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, balance_gt=False):
        super().__init__(root, transform, pre_transform)
        processed, self.extras, self.old_extras = self._load_dataset(root, num_test=num_test, num_val=num_val, balance_gt=balance_gt)
        self.data, self.slices = self.collate([processed])
        
    @property
    def num_classes(self) -> int:
        return 2
    
    def _load_dataset(self, root, num_test=0.3, num_val=0.2, balance_gt=False):
        graph_data = graph_utils.load_graph(root)
        graph_nodes, edges, has_public, has_isolates, pruning, extras = graph_data
        old_extras = extras.copy()
        edge_type_nodes = {
            'apex': ['domain', 'domain'],
            'similar': ['domain', 'domain'],
            'resolves': ['domain', 'ip'],
            'in_asn': ['ip', 'asn'],
        }

        data = HeteroData()

        node_ilocs = {}
        for node_type, node_features in graph_nodes.items():
            node_type = node_type + "_node"
            for x, idx in enumerate(node_features.index):
                node_ilocs[idx] = (x, node_type)

            data[node_type].num_nodes = node_features.shape[0]
            data[node_type].x = torch.from_numpy(node_features.values).float()
            if node_type == 'domain_node':
                extras = pd.DataFrame(extras[extras.node_id.isin(node_features.index)])
                extras['node_iloc'] = extras['node_id'].apply(lambda x: node_ilocs[x][0] if x in node_ilocs else None)
                extras = extras.dropna()
                
                labels = extras.sort_values('node_iloc')['type'].apply(lambda x: 1 if x == 'malicious' else (0 if x == 'benign' else 2))
                
                data[node_type].y = torch.from_numpy(labels.values)
                labeled = labels.values < 2
                labeled_indices = labeled.nonzero()[0]
                
                
                # balance benign and mal nodes
                if balance_gt:
                    mal_nodes = (labels.values == 1).nonzero()[0]
                    ben_nodes = (labels.values == 0).nonzero()[0]

                    min_count = min(len(mal_nodes), len(ben_nodes))
                    perm = torch.randperm(min_count)

                    mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
                    ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
                    labeled_indices = np.concatenate((mal_nodes, ben_nodes))
                
                n_nodes = len(labeled_indices)
                perm = torch.randperm(n_nodes)

                test_idx = labeled_indices[perm[:int(n_nodes * num_test)]]
                val_idx = labeled_indices[perm[int(n_nodes * num_test):int(n_nodes * (num_test + num_val))]]
                train_idx = labeled_indices[perm[int(n_nodes * (num_test + num_val)):]]

                for v, idx in [('train', train_idx), ('test', test_idx), ('val', val_idx)]:
                    mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
                    mask[idx] = True
                    data[node_type][f'{v}_mask'] = mask
        
        old_extras['node_iloc'] = old_extras['node_id'].apply(lambda x: node_ilocs[x][0] if x in node_ilocs else None)

        for edge_type, edge_data in edges.groupby('type'):
            from_type = edge_data['source'].apply(lambda x: node_ilocs[x][1]).drop_duplicates().values[0]
            to_type = edge_data['target'].apply(lambda x: node_ilocs[x][1]).drop_duplicates().values[0]

            edge_data['source'] = edge_data['source'].apply(lambda x: node_ilocs[x][0])
            edge_data['target'] = edge_data['target'].apply(lambda x: node_ilocs[x][0])
            edge_data = torch.from_numpy(edge_data.loc[:, ['source', 'target']].values.T)

            data[from_type, edge_type, to_type].edge_index = edge_data
            
        return data, extras, old_extras
    
    def to_homogeneous(self, transform=None):
        return to_homogeneous(self.data, transform=transform)
    
    @staticmethod
    def get_edge_map(data):
        if 'edge_map' in data:
            return [(
                str(edge_type),
                (data.node_type == edge_node_types[0]).nonzero().view(1, -1).cpu().numpy()[0],
                (data.node_type == edge_node_types[1]).nonzero().view(1, -1).cpu().numpy()[0]
            ) for edge_type, edge_node_types in data.edge_map.items()]
        else:
            return None
    

class HeteroDataset(Dataset):
    def __init__(self):
        pass
    
    @classmethod
    def from_storage(self, root: str, name: str = 'DNS', seed: int = 42, is_hetero: bool = True):
        graph_data = graph_utils.load_graph(root)
        data = graph_utils.preprocess_graph(graph_data, split_features=False, seed=seed, get_rel_nodes=is_hetero)
        
        adj, features, y, idx_labelled, idx_train, idx_val, idx_test, *idx_masks = data
        labels = np.array(np.argmax(y, axis=-1)).squeeze()

        if not is_hetero:
            adj = functools.reduce(lambda a, b: np.add(a, b), adj)
            
        self.name = name
        self.adj = adj
        self.features = features
        self.labels = labels
        
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.idx_labelled = idx_labelled
        if len(idx_masks) == 1:
            self.idx_masks = idx_masks[0]
        return self
            
    @classmethod
    def from_pyg(cls, pyg_data: HomoDNS, name: str = 'DNS', is_hetero=True):
        def mask_to_index(index, size):
            all_idx = np.arange(size)
            return all_idx[index]

        data = cls()
        data.name = name
        
        pyg_data = pyg_data.cpu()
        n = pyg_data.x.size(0)
        if is_hetero and 'edge_type' in pyg_data:
            adjs = []
            for i in range(int(pyg_data.edge_type.max()) + 1):
                tmp = masked_edge_index(pyg_data.edge_index, pyg_data.edge_type == i)
                
                adjs.append(sp.csr_matrix((np.ones(tmp.shape[1]), (tmp[0], tmp[1])), shape=(n, n)))
            data.adj = adjs
        else:
            data.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
                (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        
        data.features = pyg_data.x.numpy()
        data.labels = pyg_data.y.numpy()
        
        try:
            data.idx_train = mask_to_index(pyg_data.train_mask, n)
            data.idx_val = mask_to_index(pyg_data.val_mask, n)
            data.idx_test = mask_to_index(pyg_data.test_mask, n)
        except AttributeError:
            print('Warning: This pyg dataset is not associated with any data splits...')
            
        if 'edge_map' in pyg_data:
            data.idx_masks = [(
                str(edge_type),
                (pyg_data.node_type == edge_node_types[0]).nonzero().view(1, -1).cpu().numpy()[0],
                (pyg_data.node_type == edge_node_types[1]).nonzero().view(1, -1).cpu().numpy()[0]
            ) for edge_type, edge_node_types in pyg_data.edge_map.items()]
        
        return data
    
    def __repr__(self):
        if isinstance(self.adj, list):
            return '{0}(adj_count={1}, adj_shape={2}, feature_shape={3})'.format(
                self.name,
                len(self.adj),
                self.adj[0].shape if len(self.adj) > 0 else -1, 
                self.features.shape
            )
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)


def to_homogeneous(data, head_node=None, transform=None, with_mapping=True, concat_features=True):
    homo_data = data.clone()
    device = 'cuda' if homo_data.is_cuda else 'cpu'

    features_shape = {nt: node_features.shape[1] for nt, node_features in homo_data.x_dict.items()}
    if concat_features:
        if all([list(features_shape.values())[0] == v for v in features_shape.values()]):
            print("Feature dimentions are equal! Not concatinating")
            concat_features = False
        else:
            features_shape = sum([v for v in features_shape.values()])
        
    mask_types = ['train_mask', 'val_mask', 'test_mask']
    if head_node is not None:
        mask_types = [k for k in list(data[head_node].keys()) if 'mask' in k]
    masks = {k: [] for k in mask_types}
    y = []

    if with_mapping:
        node_map = {node_type: i for i, node_type in enumerate(homo_data.node_types)}
        edge_map = {i: (node_map[edge_type[0]], node_map[edge_type[2]]) for i, edge_type in enumerate(homo_data.edge_types)}

    l_padding, feat_list = 0, []
    for node_type, node_features in homo_data.x_dict.items():
        if 'y' in homo_data[node_type]:
            y.append(homo_data[node_type].y)
            for mask_type in mask_types:
                masks[mask_type].append(homo_data[node_type][mask_type])
        else:
            y.append(torch.zeros(node_features.shape[0], dtype=torch.bool).to(device))
            for mask_type in mask_types:
                masks[mask_type].append(torch.zeros(node_features.shape[0], dtype=torch.bool).to(device))

        if concat_features:
            node_features = node_features.cpu().numpy()
            r_padding = features_shape - node_features.shape[1] - l_padding
            features = []
            for node_feature in node_features:
                resized = np.pad(node_feature, (l_padding, r_padding), 'constant', constant_values=(0, 0))
                features.append(resized)

            l_padding += node_features.shape[1]
            homo_data[node_type].x =  torch.from_numpy(np.array(features)).float()
        else:
            feat_list.append(node_features)

    homo_data = homo_data.to_homogeneous(add_edge_type=True, add_node_type=True)

    for mask_type, mask in masks.items():
        homo_data[mask_type] = torch.cat(mask)
    homo_data.y = torch.cat(y)
    num_nodes = homo_data.num_nodes

    if transform is not None:
        transform(homo_data)
        
    if with_mapping:
        homo_data.edge_map = edge_map
        homo_data.num_nodes = num_nodes
        
    if (not concat_features) and 'x' not in homo_data:
        homo_data.x = feat_list

    return homo_data


def get_edge_map(data):
    if 'edge_map' in data:
        return [(
            str(edge_type),
            (data.node_type == edge_node_types[0]).nonzero().view(1, -1).cpu().numpy()[0],
            (data.node_type == edge_node_types[1]).nonzero().view(1, -1).cpu().numpy()[0]
        ) for edge_type, edge_node_types in data.edge_map.items()]
    else:
        return None


def load_dataset_homo(data_dir, graph_name, device):
    knowledge_graph_path = f'{data_dir}/{graph_name}'

    if 'mh' in graph_name or 'dns' in graph_name or 'other' in graph_name:
        dataset = DNS(root=knowledge_graph_path, transform=T.NormalizeFeatures())
        data, num_classes = dataset.to_homogeneous(transform=T.ToUndirected()).to(device), dataset.num_classes
    elif graph_name == 'imdb':
        dataset = IMDB(knowledge_graph_path)
        data, num_classes  = dataset[0], int(dataset[0]['movie'].y.max()) + 1
        del data[('director', 'to', 'movie')]
        del data[('actor', 'to', 'movie')]
        data = to_homogeneous(data, transform=T.ToUndirected()).to(device)
    elif graph_name == 'dblp':
        dataset = DBLP(knowledge_graph_path)
        data, num_classes = dataset[0], int(dataset[0]['author'].y.max()) + 1
        data['conference'].x = torch.eye(data['conference'].num_nodes, dtype=torch.float)
        del data[('paper', 'to', 'author')]
        del data[('paper', 'to', 'conference')]
        del data[('term', 'to', 'paper')]
        data = to_homogeneous(data, transform=T.ToUndirected()).to(device)
    else:
        raise NotImplemented()

    return data, num_classes, None


def load_dataset_hete(data_dir, graph_name):
    knowledge_graph_path = f'{data_dir}/{graph_name}'

    if 'dns' in graph_name:
        dataset = DNS(root=knowledge_graph_path, transform=T.Compose([T.NormalizeFeatures(), T.ToUndirected()]))
        data, num_classes, head_node = dataset[0], dataset.num_classes, 'domain_node'
    elif graph_name == 'imdb':
        dataset = IMDB(knowledge_graph_path)
        data, num_classes, head_node = dataset[0], int(dataset[0]['movie'].y.max()) + 1, 'movie'
    elif graph_name == 'dblp':
        dataset = DBLP(knowledge_graph_path)
        data, num_classes, head_node = dataset[0], int(dataset[0]['author'].y.max()) + 1, 'author'
        data['conference'].x = torch.eye(data['conference'].num_nodes, dtype=torch.float)
    else:
        raise NotImplemented()
        
    return data, num_classes, head_node