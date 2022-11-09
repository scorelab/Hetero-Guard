import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import scipy.sparse as sps
from sklearn import model_selection, preprocessing


def to_networkx(
        G,
        node_type_attr='type',
        edge_type_attr='type',
        edge_weight_attr='weight',
        feature_attr=None,
        reset_indices=False
    ):
        """
        Create a NetworkX Graph instance representing this graph.
        Args:
            node_type_attr (str): the name of the attribute to use to store a node's type (or label).
            edge_type_attr (str): the name of the attribute to use to store a edge's type (or label).
            edge_weight_attr (str): the name of the attribute to use to store a edge's weight.
            feature_attr (str, optional): the name of the attribute to use to store a node's feature
                vector; if ``None``, feature vectors are not stored within each node.
            reset_indices (boolean): whether to reset the node ids or not.
        Returns:
             An instance of `networkx.Graph` containing all the nodes & edges and their types & 
             features in this graph.
        """

        graph = nx.Graph()
        
        if reset_indices:
            all_nodes = []
            for ty in G.node_types:
                all_nodes.extend(G.nodes(node_type=ty))
                
            idx_map = {node_id: idx for idx, node_id in enumerate(all_nodes)}

        for ty in G.node_types:
            node_ids = G.nodes(node_type=ty)
            ty_dict = {node_type_attr: ty}
            features = G.node_features(node_ids, node_type=ty)

            if reset_indices:
                node_ids = [idx_map[x] for x in node_ids]

            if feature_attr is not None:
                for node_id, node_features in zip(node_ids, features):
                    graph.add_node(
                        node_id, **ty_dict, **{feature_attr: node_features},
                    )
            else:
                graph.add_nodes_from(node_ids, **ty_dict)
                
        iterator = zip(
            [idx_map[x] for x in G._nodes.ids.from_iloc(G._edges.sources)] if reset_indices else G._nodes.ids.from_iloc(G._edges.sources),
            [idx_map[x] for x in G._nodes.ids.from_iloc(G._edges.targets)] if reset_indices else G._nodes.ids.from_iloc(G._edges.targets),
            G._edges.type_of_iloc(slice(None)),
            G._edges.weights,
        )
            
        graph.add_edges_from(
            (src, dst, {edge_type_attr: type_, edge_weight_attr: weight})
            for src, dst, type_, weight in iterator
        )

        if reset_indices:
            return graph, idx_map
        return graph
    

def to_gexf(G_nx, extra, path, node_type_attr='type', edge_type_attr='type', edge_weight_attr='weight', label_attr='type'):
    temp_G = nx.Graph()

    node_list = []
    for node_id, node_data in G_nx.nodes(data=True):
        value = extra.loc[node_id, 'value'] if node_id in extra.index else ""
        suffix = ("_" + extra.loc[node_id, label_attr]) if node_id in extra.index else ""
        node_attr = {
            'label': value,
            node_type_attr: node_data[node_type_attr] + suffix
        }
        node_list.append((node_id, node_attr))

    temp_G.add_nodes_from(node_list)
        
    temp_G.add_edges_from(
        (
            int(row['source']), 
            int(row['target']), 
            {edge_type_attr: int(row[edge_type_attr]), edge_weight_attr: row[edge_weight_attr]}
        ) for idx, row in nx.to_pandas_edgelist(G_nx).iterrows()
    )
        
    nx.write_gexf(temp_G, path=path)    
    

def plot_degree_dist_log(G, title=""):
    G_degrees = pd.DataFrame(G.node_degrees().items(), columns=['node', 'degree']).sort_values('degree', ascending=False)
    G_degrees = G_degrees.degree.values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    
    [counts,bins,patches]=ax[0].hist(G_degrees,bins=G_degrees[0], color='b', lw=0)
    ax[0].set_xlabel('degreee')
    ax[0].set_ylabel('frequency')

    countsnozero=counts*1.
    countsnozero[counts==0]=-sp.Inf

    ax[1].scatter(bins[:-1],countsnozero/float(sum(counts)),s=60)
    ax[1].set_yscale('log'), ax[1].set_xscale('log')
    ax[1].set_ylim(0.00008,1.1), ax[1].set_xlim(0.8,1100)
    ax[1].set_xlabel('degree')
    ax[1].set_ylabel("fraction of nodes")
#     plt.subplots_adjust(bottom=0.15)
    plt.title(title)
    plt.show()
    

def plot_degree_dist_nx(g):
    degrees = nx.degree(g)
    degree_df = pd.DataFrame(degrees, columns=['node_id', 'degree']).sort_values('degree', ascending=False)   
    g_degrees = degree_df.degree.values
    print(g_degrees)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))

    [counts,bins,patches]=ax[0].hist(g_degrees,bins=g_degrees[0], color='b', lw=0)
    ax[0].set_xlabel('Degree')
    ax[0].set_ylabel('Frequency')

    countsnozero=counts*1.
    countsnozero[counts==0]=-np.Inf

    ax[1].scatter(bins[:-1],countsnozero/float(sum(counts)),s=60)
    ax[1].set_yscale('log'), ax[1].set_xscale('log')
    ax[1].set_ylim(0.00008,1.1), ax[1].set_xlim(0.8,1100)
    ax[1].set_xlabel('Degree (log)')
    ax[1].set_ylabel("Frequency (log)")
    plt.show()
    

def load_graph(path, not_multigraph=True, split_extras=False):
    """
    Load stored graph in the given directory 

    :param path: Directory of the stored graph
    :param split_extras: Whether to load graph extras as a seperate dataset instead of within the graph
    :returns: KnowledgeGraph instance (tuple of KnowledgeGraph instance and DataFrame of extras if split_extras=True)
    """
    with open(os.path.join(path, 'summary.json'), 'r') as json_data:
        summary = json.load(json_data)

    has_public = summary['has_public']
    has_isolates = summary['has_isolates']
    pruning = summary['pruning']

    directory = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in directory if os.path.isfile(f)]

    edges = [f for f in files if 'edges' in f]
    if len(edges) > 0:
        edges = pd.read_csv(edges[0])
    else:
        raise Exception("No 'edges.csv' file found in the path")

    if not_multigraph:
        edges_sorted = pd.DataFrame(np.sort(edges.loc[:, ['source', 'target']].values, axis=1), columns=['source', 'target'])
        edges_sorted['type'] = edges.type

        edges_sorted = edges_sorted.sort_values(['source', 'target', 'type'])
        print(f"Remove parallel edges: {edges_sorted[edges_sorted.duplicated(['source', 'target'])].value_counts('type')}")
        edges = edges_sorted.drop_duplicates(['source', 'target'])

    graph_nodes = {}
    nodes = [f for f in files if 'nodes' in f]
    if len(nodes) > 0:
        for n_type_file in nodes:
            n_type = n_type_file.split('.')[-2]
            nodes_df = pd.read_csv(n_type_file, index_col=0)
            graph_nodes[n_type] = nodes_df
    else:
        raise Exception("No 'nodes.<node_type>.csv' files found in the path")

    extras = [f for f in files if 'extras' in f]
    if len(extras) > 0:
        extras = pd.read_csv(extras[0], index_col=0)
    else:
        extras = None

    return graph_nodes, edges, has_public, has_isolates, pruning, extras


def get_node_ids(node_features, node_type):
    nodes_df = node_features.reset_index().loc[:, ['node_id']]
    nodes_df['type'] = [node_type] * len(nodes_df)
    return nodes_df


def get_adjacency_matrix(node_ilocs, edges, edge_type=None):
    type_edges = edges[edges.type == edge_type]
    node_ilocs = node_ilocs.reset_index().set_index('node_id')

    src_idx = type_edges.join(node_ilocs, on='source', rsuffix='_').loc[:, 'index'].values
    tgt_idx = type_edges.join(node_ilocs, on='target', rsuffix='_').loc[:, 'index'].values
        
    n = node_ilocs.shape[0]
    weights = np.ones(src_idx.shape)

    adj = sps.csr_matrix((weights, (src_idx, tgt_idx)), shape=(n, n))
    if n > 0:
        # in an undirected graph, the adjacency matrix should be symmetric: which means counting
        # weights from either "incoming" or "outgoing" edges, but not double-counting self loops

        # FIXME https://github.com/scipy/scipy/issues/11949: these operations, particularly the
        # diagonal, don't work for an empty matrix (n == 0)
        backward = adj.transpose(copy=True)
        # this is setdiag(0), but faster, since it doesn't change the sparsity structure of the
        # matrix (https://github.com/scipy/scipy/issues/11600)
        (nonzero,) = backward.diagonal().nonzero()
        backward[nonzero, nonzero] = 0

        adj += backward

    # this is a multigraph, let's eliminate any duplicate entries
    adj.sum_duplicates()
    return adj
  

def get_rel_node_types(node_ilocs, edges, edge_type=None):
    type_edges = edges[edges.type == edge_type]
    node_ilocs = node_ilocs.reset_index().set_index('node_id')

    src = type_edges.join(node_ilocs, on='source', rsuffix='_')
    src_types = src['type_'].drop_duplicates().values

    tgt = type_edges.join(node_ilocs, on='target', rsuffix='_')
    tgt_types = tgt['type_'].drop_duplicates().values

    src_idx = node_ilocs[node_ilocs.type.isin(src_types)].loc[:, 'index'].values
    tgt_idx = node_ilocs[node_ilocs.type.isin(tgt_types)].loc[:, 'index'].values

    return (edge_type, src_idx, tgt_idx)


def preprocess_graph(graph_data, split_features=False, get_rel_nodes=False, seed=42):
    graph_nodes, edges, has_public, has_isolates, pruning, extras = graph_data
    
     # Hack to get the domains first (because of labelling)
    graph_nodes_items = sorted(graph_nodes.items(), key=lambda item: '_' if item[0] == 'domain' else item[0])

    node_ilocs = pd.concat([get_node_ids(node_features, node_type) for node_type, node_features in graph_nodes_items]).reset_index(drop=True)
    
    domain_node_ids = node_ilocs[node_ilocs.type == 'domain'].loc[:, ['node_id']]
    labelled_with_na = domain_node_ids.join(extras.set_index("node_id"), on='node_id').drop(columns=['value']).reset_index()
    labelled_nodes = labelled_with_na[(labelled_with_na.type == 'benign') | (labelled_with_na.type == 'malicious')]

    node_classes = labelled_nodes.loc[:, 'type']
    train_nodes, test_nodes = model_selection.train_test_split(
        labelled_nodes, train_size=0.5, stratify=node_classes, random_state=seed
    )
    val_nodes, test_nodes = model_selection.train_test_split(
        test_nodes, train_size=0.2, stratify=test_nodes['type'], random_state=seed
    )
    
    adjs = []
    rel_nodes = []
    for edge_type in edges['type'].drop_duplicates().values:
        adjs.append(get_adjacency_matrix(node_ilocs, edges, edge_type=edge_type))
        if get_rel_nodes:
            rel_nodes.append(get_rel_node_types(node_ilocs, edges, edge_type=edge_type))

    features = []
    features_shape = sum([node_features.shape[1] for _, node_features in graph_nodes_items])
    l_padding = 0
    for node_type, node_features in graph_nodes_items:
        node_features = node_features.values
        if not split_features:
            r_padding = features_shape - node_features.shape[1] - l_padding
            for node_feature in node_features:
                resized = np.pad(node_feature, (l_padding, r_padding), 'constant', constant_values=(0, 0))
                features.append(resized)

            l_padding += node_features.shape[1]
        else:
            features.extends(node_features)
    features = np.array(features)

    one_hot_encoding = preprocessing.OneHotEncoder()
    labels = one_hot_encoding.fit_transform(labelled_with_na.loc[:, 'type'].values.reshape(-1, 1)).toarray()
    idx_labelled = labelled_nodes.loc[:, 'index'].values
    
    if get_rel_nodes:
        return adjs, features, labels, idx_labelled, train_nodes['index'].values, val_nodes['index'].values, test_nodes['index'].values, rel_nodes
    return adjs, features, labels, idx_labelled, train_nodes['index'].values, val_nodes['index'].values, test_nodes['index'].values


def preprocess_aifb(data):
    import torch
    data.x = torch.ones((data.num_nodes, 1))

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[data.train_idx] = True

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.test_idx] = True

    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[data.test_idx] = True

    data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    data.y[data.train_idx] = data.train_y
    data.y[data.test_idx] = data.test_y

    data.edge_type = torch.tensor([int(et / 2) for et in data.edge_type])

    data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    data.edge_map = {i: (0, 0) for i in data.edge_type.unique().numpy()}
    return data


# Legacy 
def preprocess_stellargraph(G):
    domain_node_ids = G.nodes(node_type='domain')
    labelled_with_na = domain_node_ids.join(G.extras.set_index("node_id"), on='node_id').drop(columns=['value']).reset_index()
    labelled_nodes = labelled_with_na[(labelled_with_na.type == 'benign') | (labelled_with_na.type == 'malicious')]

    node_classes = labelled_nodes.loc[:, 'type']
    train_nodes, test_nodes = model_selection.train_test_split(
        labelled_nodes, train_size=0.5, stratify=node_classes, random_state=42
    )
    val_nodes, test_nodes = model_selection.train_test_split(
        test_nodes, train_size=0.3, stratify=test_nodes['type'], random_state=42
    )
    
    one_hot_encoding = preprocessing.OneHotEncoder()
    one_hot_encoding.fit(labelled_nodes.loc[:, 'type'].values.reshape(-1, 1))

    y_train = one_hot_encoding.transform(train_nodes.loc[:, 'type'].values.reshape(-1, 1)).toarray()
    y_val = one_hot_encoding.transform(val_nodes.loc[:, 'type'].values.reshape(-1, 1)).toarray()
    y_test = one_hot_encoding.transform(test_nodes.loc[:, 'type'].values.reshape(-1, 1)).toarray()

    return train_nodes.node_id.values, y_train, val_nodes.node_id.values, y_val, test_nodes.node_id.values, y_test