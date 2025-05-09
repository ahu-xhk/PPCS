import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

def load_data(dataset_name):

    communties = open(f"../data/{dataset_name}/{dataset_name}-1.90.cmty.txt")
    edges = open(f"../data/{dataset_name}/{dataset_name}-1.90.ungraph.txt")

    communties = [[int(i) for i in x.split()] for x in communties]
    edges = [[int(i) for i in e.split()] for e in edges]
    # Remove self-loop edges
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]

    num_node, num_edges, num_comm = len(nodes), len(edges), len(communties)
    print(f"[{dataset_name.upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")

    node_feats = None

    return num_node, num_edges, num_comm, nodes, edges, communties, node_feats


def feature_augmentation(nodes, edges, num_node, normalize=True, feat_type='AUG'):
    if feat_type == "ONE":
        return np.ones([num_node, 1], dtype=np.float32)

    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    for node in range(num_node):
        if len(list(g.neighbors(node))) > 0:
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)
    return feat_matrix, g


def prepare_data(dataset="amazon"):
    num_node, num_edge, num_community, nodes, edges, communities, features = load_data(dataset)
    if features is None:
        features, nx_graph = feature_augmentation(nodes, edges, num_node)
    else:
        nx_graph = nx.Graph(edges)
        nx_graph.add_nodes_from(nodes)


    converted_edges = [[v, u] for u, v in edges]

    graph_data = Data(x=torch.FloatTensor(features), edge_index=torch.LongTensor(np.array(edges+converted_edges)).transpose(0, 1))

    return num_node, num_edge, num_community, graph_data, nx_graph, communities
