from typing import Union, Optional, List
import numpy as np
from scipy import sparse as sp

from .graph import Graph

class ExpansionEnv:

    def __init__(self,
                 graph: Graph,
                 selected_nodes: List[List[int]],
                 max_size: int,
                 embeddings
                 ):
        self.max_size = max_size     
        self.graph = graph        
        self.n_nodes = self.graph.n_nodes  
        self.data = selected_nodes  
        self.bs = len(self.data)     
        self.trajectories = None   
        self.dones = None            
        self.embeddings = embeddings


    @property
    def lengths(self):
        return [len(x) - (x[-1] == 'EOS') for x in self.trajectories]

    @property
    def done(self):
        return all(self.dones)

    @property
    def valid_index(self) -> List[int]:
        return [i for i, d in enumerate(self.dones) if not d]


    def __len__(self):
        return len(self.data)

    def updateGraph(self, graph):

        self.graph = graph
        self.n_nodes = self.graph.n_nodes

    def reset(self):

        self.trajectories = [x.copy() for x in self.data]
        self.dones = [x[-1] == 'EOS' or len(x) >= self.max_size or len(self.graph.outer_boundary(x)) == 0
                      for x in self.trajectories]
        assert not any(self.dones) 
 
        seeds = [self.data[i][0] for i in range(self.bs)]
    
        nodes = [self.data[i] for i in range(self.bs)]

        x_seeds = self.make_single_node_encoding(seeds)
        x_nodes = self.make_nodes_encoding(nodes)

        return x_seeds, x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int]):

        assert len(new_nodes) == len(index)
        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            self.trajectories[i].append(v)
            if v == 'EOS':
                self.dones[i] = True
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif self.graph.outer_boundary(self.trajectories[i]) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v
        delta_x_nodes = self.make_single_node_encoding(full_new_nodes)

        return delta_x_nodes



    def make_single_node_encoding(self, nodes: List[int]):

        embeddings = self.embeddings
        embeddings_dim = 128

        bs = len(nodes)
        ind = np.array([[v, i] for i, v in enumerate(nodes) if v is not None], dtype=np.int64).T
        if len(ind):
            data = [np.max(embeddings.get(v, np.zeros(embeddings_dim))) for v in ind[0]]
            data = np.array(data, dtype=np.float32)
            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)




    def make_nodes_encoding(self, nodes: List[List[int]]):

        embeddings = self.embeddings
        embeddings_dim = 128

 
        bs = len(nodes)
        assert bs == self.bs
        ind = [[v, i] for i, vs in enumerate(nodes) for v in vs]
        ind = np.asarray(ind, dtype=np.int64).T
        if len(ind):

            data = [np.max(embeddings.get(v, np.zeros(embeddings_dim))) for v in ind[0]]
            data = np.array(data, dtype=np.float32)
            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)

