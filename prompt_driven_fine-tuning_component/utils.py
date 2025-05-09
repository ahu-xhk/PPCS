import numpy as np
import random
import torch
import datetime
import time
import pytz
from torch_geometric.utils import k_hop_subgraph

def set_seed(seed): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def generate_prompt_tuning_data(train_comm, graph_data, nx_graph, k=2):

    degrees = [nx_graph.degree[node] for node in train_comm]
    sum_val = sum(degrees)
    degrees = [d / sum_val for d in degrees]

    central_node = np.random.choice(train_comm, 1, p=degrees).tolist()[0]

    k_ego_net, _, _, _ = k_hop_subgraph(central_node, num_hops=k, edge_index=graph_data.edge_index,
                                        num_nodes=graph_data.x.size(0))
    k_ego_net = k_ego_net.detach().cpu().numpy().tolist()
    labels = [[int(node in train_comm)] for node in k_ego_net]

    if 0 not in labels:

        random_negatives = np.random.choice(graph_data.x.size(0), len(k_ego_net)).tolist()
        random_negatives = [node for node in random_negatives if node not in k_ego_net]  # remove positive samples
        k_ego_net += random_negatives
        labels += [[0]] * len(random_negatives)

    return [central_node] * len(k_ego_net), k_ego_net, torch.FloatTensor(labels)
