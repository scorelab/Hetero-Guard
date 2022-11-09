import json


def load_config(dataset="mDNS", gnn="HeteroSAGE", attack=None, config_file='./config.json'):
    """
    Load configuration for particular dataset/GNN/attack

    :param dataset: Name of the dataset (mDNS|IMDB|DBLP)
    :param gnn: Name of the GNN (GCN|HeteroSAGE|HAN)
    :param attack: Name of the attack (PRBCD|ConstPRBCD|HetePRBCD|ConstPRBCD)
    :param config_file: Path to the configuration file
    :return:
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    if attack is None:
        return config['gnn'][dataset][gnn]
    else:
        return config['gnn'][dataset][gnn], config['attack'][dataset][attack]