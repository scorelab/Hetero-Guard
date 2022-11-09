# HeteroSAGE HetePRBCD

import argparse
import torch

from libs.attack import const_hete_prbcd, const_prbcd
from libs.loader import load_dataset_hete, load_dataset_homo
from libs.gnn import HeteroSAGE, GCN
from libs.config import load_config
from datetime import datetime


gnn_models = {
    'GCN': GCN,
    "HeteroSAGE": HeteroSAGE,
}


def attack(db_name, gnn_name, attack_name, perturbation_rate, constraint, biased, data_dir, cuda_device):
    print(f"Cuda Available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        device = 'cuda'
    else:
        device = 'cpu'

    gnn_config, atk_config = load_config(dataset=db_name, gnn=gnn_name, attack=attack_name)
    is_hete = 'is_hete' in gnn_config and gnn_config['is_hete']
    no_layers = gnn_config['num_layers']
    no_hidden = gnn_config['num_hidden']
    pert_sample = atk_config['block_size']
    lamb = atk_config['lambda'] if biased else None

    load_fn = load_dataset_hete if is_hete else load_dataset_homo
    data, num_classes, head_node = load_fn(data_dir, db_name)

    # Train vitcim model
    gnn_class = gnn_models[gnn_name]
    if is_hete:
        victim = gnn_class(data, no_layers, no_hidden, num_classes, head_node, dropout=0.5)
    else:
        victim = gnn_class(data, no_layers, no_hidden, num_classes, dropout=0.5)
    history = victim.fit()

    # Define perturbation rate
    if constraint:
        if db_name in ['mdns']:
            constraints = [('domain_node', 'resolves', 'ip_node')]
            hete_sym = {('domain_node', 'resolves', 'ip_node'): ('ip_node', 'rev_resolves', 'domain_node')}
        elif db_name == 'dblp':
            constraints = [('author', 'to', 'paper')]
            hete_sym = {('author', 'to', 'paper'): ('paper', 'to', 'author')}
        elif db_name == 'imdb':
            constraints = [('movie', 'to', 'actor')]
            hete_sym = {('movie', 'to', 'actor'): ('actor', 'to', 'movie')}
    else:
        if db_name in ['mdns']:
            constraints = [('domain_node', 'apex', 'domain_node'), ('domain_node', 'similar', 'domain_node'),
                           ('domain_node', 'resolves', 'ip_node')]
            hete_sym = {('domain_node', 'resolves', 'ip_node'): ('ip_node', 'rev_resolves', 'domain_node')}
        elif db_name == 'dblp':
            constraints = [('author', 'to', 'paper'), ('paper', 'to', 'term'), ('paper', 'to', 'conference')]
            hete_sym = {('author', 'to', 'paper'): ('paper', 'to', 'author'),
                        ('paper', 'to', 'term'): ('term', 'to', 'paper'),
                        ('paper', 'to', 'conference'): ('conference', 'to', 'paper')}
        elif db_name == 'imdb':
            constraints = [('movie', 'to', 'actor'), ('movie', 'to', 'director')]
            hete_sym = {('movie', 'to', 'actor'): ('actor', 'to', 'movie'),
                        ('movie', 'to', 'director'): ('director', 'to', 'movie')}

    n_perturbations = int((perturbation_rate / 100) * (data.num_edges / 2))
    print(n_perturbations)

    # Attack
    model = const_hete_prbcd.ConstHetePRBCD(
        victim,
        data,
        pert_sample,
        head_node=head_node,
        budget=constraints,
        hete_symmetric=hete_sym,
        lamb=lamb,
        loss_type='CE',
        epochs=200
    )
    attack_history = model.attack(n_perturbations, check_modified=False)
    modified_data = model.modified

    # Train GCN on attacked graph
    attacked = HeteroSAGE(modified_data, no_layers, no_hidden, num_classes, head_node, dropout=0.5, weight_decay=5e-4)
    history_attacked = attacked.fit()

    return {
        'random_sample_size': pert_sample,
        'lambda': lamb,
        'no_layers': no_layers,
        'no_hidden': no_hidden,
        'victim': victim.test(),
        'attacked': attacked.test(),
        'timestamp': str(datetime.now())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_name", help="Dataset name", type=str)
    parser.add_argument("gnn_name", help="GNN name", type=str)
    parser.add_argument("attack_name", help="Attack name", type=str)
    parser.add_argument("perturb", help="Perturbation percentange", type=int)
    parser.add_argument("data_dir", help="Directory of data", type=str)
    parser.add_argument("-c", "--constrained", help="Run constrained", action="store_true")
    parser.add_argument("-b", "--biased", help="Run biased", action="store_true")
    parser.add_argument("--cuda", help="Cuda device", type=int, default=2)
    args = parser.parse_args()

    perturbation_rate = args.perturb
    db_name = args.db_name
    gnn_name = args.gnn_name
    attack_name = args.attack_name
    data_dir = args.data_dir
    constraint = args.constrained
    biased = args.biased
    device = args.cuda
    print("On CUDA:", device)

    print("Start attack!")
    attack_stats = attack(db_name, gnn_name, attack_name, perturbation_rate, constraint, biased, data_dir, device)
    print("RESULTS!!!")
    print("Accuracy drop:", attack_stats['victim']['acc']['test'] - attack_stats['attacked']['acc']['test'])
    print("Attack Done!")
