from typing import Union, Optional, List, Set, Dict
import collections
import numpy as np
from scipy import sparse as sp
import random


class Graph:

    def __init__(self, edges):

        self.neighbors, self.n_nodes, self.adj_mat, self.degree = self._init_from_edges(edges)

    @staticmethod
    def _init_from_edges(edges: np.ndarray) -> (Dict[int, Set[int]], int, sp.spmatrix, Dict[int, int]):
        neighbors = collections.defaultdict(set)
        degrees = {}
        max_id = -1
        for u, v in edges:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1

            max_id = max(max_id, u, v)
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
        n_nodes = len(neighbors)
        if (max_id + 1) != n_nodes:
            raise ValueError('Please re-label nodes first!')
        adj_mat = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(n_nodes, n_nodes))
        adj_mat += adj_mat.T
        return neighbors, n_nodes, adj_mat, degrees

    def setParentGraph(self, parentGraph):

        self.parentGraph = parentGraph

    def outer_boundary(self, nodes: Union[List, Set]) -> Set[int]:
        boundary = set()
        for u in nodes:
            boundary |= self.neighbors[u]  
        boundary.difference_update(nodes)
        return boundary

    def k_ego(self, nodes: Union[List, Set], k: int) -> Set[int]:

        ego_nodes = set(nodes)
        current_boundary = set(nodes)
        for _ in range(k):
            current_boundary = self.outer_boundary(current_boundary) - ego_nodes
            ego_nodes |= current_boundary
        return ego_nodes

    def get_k_layer_subgraph_and_mapping(self, node_list: Union[List[int], Set[int]], k: int):
 
        k_layer_neighbors = self.k_ego(node_list, k)

        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(k_layer_neighbors))}

        subgraph_edges = []

        for node in k_layer_neighbors:
            for neighbor in self.neighbors[node]:
                if neighbor in k_layer_neighbors:
                    mapped_node = node_mapping[node]
                    mapped_neighbor = node_mapping[neighbor]
                    if mapped_node < mapped_neighbor:
                        subgraph_edges.append((mapped_node, mapped_neighbor))

        edges_array = np.array(subgraph_edges)


        subgraph = Graph(edges_array)

        return subgraph, node_mapping

    def sample_expansion_from_community(self, comm_nodes: Union[List, Set],
                                        seed: Optional[int] = None) -> List[int]:

        if seed is None:
            seed = random.choice(tuple(comm_nodes))

        remaining = set(comm_nodes) - {seed}    
        boundary = self.neighbors[seed].copy()
        walk = [seed]
        while len(remaining):
            try:
                candidates = tuple(boundary & remaining)  
                new_node = random.choice(candidates)
                remaining.remove(new_node)
                boundary |= self.neighbors[new_node] 
                walk.append(new_node)
            except Exception:
                return walk
        return walk


    def add_nodes_with_neighbors(self, newIDnode_nei: Dict[int, Union[List[int], Set[int]]]):

        for new_node, neighbors in newIDnode_nei.items():
            if new_node not in self.neighbors:  
                self.n_nodes += 1 
                self.neighbors[new_node] = set()  

            for neighbor in neighbors:
                self.neighbors[new_node].add(neighbor) 
                if neighbor not in self.neighbors: 
                    self.n_nodes += 1  
                    self.neighbors[neighbor] = {new_node} 
                else:
                    self.neighbors[neighbor].add(new_node)  

                self.degree[new_node] = self.degree.get(new_node, 0) + 1
                self.degree[neighbor] = self.degree.get(neighbor, 0) + 1


        edges_list = []
        for node, nbs in self.neighbors.items():
            for nb in nbs:
                if node < nb:  
                    edges_list.append((node, nb))

        edges_array = np.array(edges_list)
        self.adj_mat = sp.csr_matrix((np.ones(len(edges_list)), edges_array.T), shape=(self.n_nodes, self.n_nodes))
        self.adj_mat += self.adj_mat.T  
