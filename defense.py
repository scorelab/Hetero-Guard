import os
import sys
import argparse
import json

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric import transforms as T

from libs.loader import load_dataset_hete
from libs.attack.const_hete_prbcd import ConstHetePRBCD
from libs.loader import HeteroDataset, to_homogeneous
from libs.gnn import HeteroSAGE
from deeprobust.graph.defense import GCNJaccard, RGCN
from libs.defence.gnnguard import gcn_attack
from libs.defence import heteroguard
from libs.config import load_config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_metrics(output, labels, idx_test, nclass):
    split_pred = output.argmax(dim=-1)[idx_test].detach().cpu().numpy()
    split_y = labels[idx_test] 

    acc = accuracy_score(split_y, split_pred)
    f1_macro = f1_score(split_y, split_pred, average='macro')
    f1_micro = f1_score(split_y, split_pred, average='micro')

    auc_pred = np.exp(output[idx_test].detach().cpu().numpy()) if nclass > 2 else split_pred
    auc = roc_auc_score(split_y, auc_pred, multi_class='ovr')

    return {'acc': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'auc': auc}


def run_attack(db_name, perturbation_rate, data_dir, out_dir, cuda_device):
    print(f"Cuda Available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        device = 'cuda' 
    else:
        device ='cpu'
        
    stats = {}
    stats['perturbation_rate'] = perturbation_rate
    
    gnn_config, atk_config = load_config(dataset=db_name, gnn="HeteroSAGE", attack="HetePRBCD")
    no_layers = gnn_config['num_layers']
    no_hidden = gnn_config['num_hidden']
    rbcd_block_size = atk_config['block_size']
    lamb = atk_config['lambda']
                
    dropout = 0.5
    weight_decay=5e-4
    attack_epochs = 200
    epochs = 200
    gcn_j_threshold = 0.3
    
    data, num_classes, head_node = load_dataset_hete(data_dir, db_name)

    # Define perturbation rate
    n_perturbation = int((perturbation_rate / 100) * (data.num_edges / 2))
    print("No. of perturbations:", n_perturbation)
    
    print("No. of nodes", data.num_nodes)
    print("No. of edges", data.num_edges)
    print("No. of node types", len(data.node_types))
    print("No. of edge types", int(len(data.edge_types)/2))
    print("No. of classes", num_classes)
    
    victim = HeteroSAGE(data, no_layers, no_hidden, num_classes, head_node, dropout=dropout, weight_decay=weight_decay)
    victim.fit(epochs=epochs)
    stats['victim'] = victim.test()

    if db_name == 'dblp':
        constraints = [('author', 'to', 'paper')]
        hete_sym = {('author', 'to', 'paper'): ('paper', 'to', 'author')}
    elif db_name == 'imdb':
        constraints = [('movie', 'to', 'director')]
        hete_sym = {('movie', 'to', 'director'): ('director', 'to', 'movie')}
    elif db_name in ['mdns']:
        constraints = [('domain_node', 'resolves', 'ip_node')]
        hete_sym = {('domain_node', 'resolves', 'ip_node'): ('ip_node', 'rev_resolves', 'domain_node')}
    else:
        raise Exception("Wrong graph")
    print(constraints, hete_sym)
    model = ConstHetePRBCD(
        victim, 
        data, 
        rbcd_block_size, 
        head_node=head_node, 
        budget=constraints, 
        hete_symmetric=hete_sym,
        lamb=lamb, 
        loss_type='CE',
        epochs=attack_epochs,
    )
    model.attack(n_perturbation, check_modified=False)
    modified = model.modified 

    print("No. of hidden", no_hidden)
    print("No. of classes", num_classes)
    attacked = HeteroSAGE(modified, no_layers, no_hidden, num_classes, head_node, dropout=dropout, weight_decay=weight_decay)
    attacked.fit(epochs=epochs)
    stats['attacked'] = attacked.test()
    print(stats)
    
    attension = heteroguard.HeteroGuardAttn(data, normalize=True, exp=False, drop_bottom=True)
    
    cleaned = HeteroSAGE(modified, no_layers, no_hidden, num_classes, head_node, attension=attension, 
                         gated_attension=True, dropout=dropout, weight_decay=weight_decay)
    cleaned.fit(epochs=epochs)
    stats['HeteroGuard'] = cleaned.test()
    
    if db_name == 'dblp':
        del modified[('paper', 'to', 'author')]
        del modified[('paper', 'to', 'conference')]
        del modified[('term', 'to', 'paper')]
    elif db_name == 'imdb':
        del modified[('director', 'to', 'movie')]
        del modified[('actor', 'to', 'movie')]
    elif db_name in ['mdns']:
        del modified[('ip_node', 'rev_resolves', 'domain_node')]
    else:
        raise Exception("Wrong graph")

    modified_data = to_homogeneous(modified.cpu(), transform=T.ToUndirected()).to(device)
    data = HeteroDataset.from_pyg(modified_data, is_hetero=False)
    modified_adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test, idx_masks = data.idx_train, data.idx_val, data.idx_test, data.idx_masks

    # GCNJaccard
    gcn_j = GCNJaccard(nfeat=features.shape[1], nhid=no_hidden, nclass=num_classes, binary_feature=False,
               dropout=dropout, device=device).to(device)
    gcn_j.fit(features, modified_adj, labels, idx_train, idx_val, threshold=gcn_j_threshold, epochs=epochs, verbose=False)
    output = gcn_j.predict()
    stats['GCNJaccard'] = get_metrics(output, labels, idx_test, num_classes)
    
    # RobustGCN
    if not (db_name in ['dblp', 'dns']):
        sp_features = sp.csr_matrix(features)
        sp_adj = sp.csr_matrix(modified_adj)

        rgcn = RGCN(nnodes=modified_adj.shape[0], nfeat=features.shape[1],
                     nclass=num_classes, nhid=no_hidden, device=device).to(device)
        rgcn.fit(sp_features, sp_adj, labels, idx_train, idx_val, train_iters=200, verbose=False)
        output = rgcn.predict()
        stats['RobustGCN'] = get_metrics(output, labels, idx_test, num_classes)

    # GNNGuard
    gnng = gcn_attack.GCN_attack(nfeat=features.shape[1], nclass=num_classes, nhid=no_hidden,
                          dropout=dropout, with_relu=True, with_bias=True, weight_decay=5e-4, device=device).to(device)
    gnng.fit(features, modified_adj, labels, idx_train, train_iters=201, idx_val=idx_val,
             idx_test=idx_test, verbose=False, attention=True)
    stats['GNNGuard'] = gnng.test(idx_test)
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Graph Name", type=str)
    parser.add_argument("data_dir", help="Directory of data", type=str)
    parser.add_argument("out_dir", help="Directory of outputs", type=str)
    parser.add_argument("--perturb", help="Perturbation percentange", type=int, default=15)
    parser.add_argument("--cuda", help="Cuda device", type=int, default=2)
    args = parser.parse_args()
    
    graph = args.graph
    data_dir = args.data_dir
    out_dir = args.out_dir
    perturbation_rate = args.perturb
    device = args.cuda
    print("On CUDA:", device)
    
    print("Start Defence!")
    run_attack(graph, perturbation_rate, data_dir, out_dir, device)
    print("Attack Done!")
