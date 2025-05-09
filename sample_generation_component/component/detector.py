import os
import shutil

import networkx as nx
import torch
from torch import optim
import time
import numpy as np
import random

from .graph import Graph
from .agent import Agent
from .expander import Expander
from .eval_com  import *


class Detector:
    def __init__(self, args, seed, com_index):
        self.args = args
        self.graph, self.coms = self.loadDataset(args.root, args.dataset)
        print(f"num_ground_truth_com：{len(self.coms)}")
        print("num_seeds：", len(seed), "\nnum_seeds_coms：", len(com_index))

        self.oldKnowcoms = random.sample(self.coms, args.train_size)

        filename = f"datasets/train_coms/{args.dataset}_train_coms.txt"
        with open(filename, 'w') as file:
            for community in self.oldKnowcoms:
                community_str = ','.join(map(str, community)) 
                file.write(community_str + '\n')
        exit()

        self.oldSeed = seed

        self.com_index = com_index

        if args.dataset == "twitter":
            fileedge = f"datasets/{self.args.dataset}/{self.args.dataset}-1.90.ungraph.txt"
            G = self.networkx(fileedge)

            self.oldKnowcoms = self.remove_disconnected_communities(self.oldKnowcoms, G)

        knowcomSeed_nodes = set([node for com in self.oldKnowcoms.copy() for node in com] + seed)  

        self.knowcomSeedGraph, self.old_to_new_node_mapping = self.graph.get_k_layer_subgraph_and_mapping(
            knowcomSeed_nodes, args.k_ego_subG)

        self.knowcomSeedGraph.setParentGraph(self.graph)

        self.new_to_old_node_mapping = {new_id: old_id for old_id, new_id in self.old_to_new_node_mapping.items()}

        self.args.old_to_new_node_mapping = self.old_to_new_node_mapping
        self.args.new_to_old_node_mapping = self.new_to_old_node_mapping


        self.knowcoms = [[self.old_to_new_node_mapping[node] for node in coms] for coms in self.oldKnowcoms]
        self.train_comms = self.knowcoms

        self.args.max_size = max(len(x) for x in self.knowcoms)

        self.seed = []
        for i in seed:
            self.seed.append(self.old_to_new_node_mapping[i])

        self.device = torch.device('cpu')
        self.expander = self.init_expander()


        self.args.nodes_embeddings = self.loadEmbeddings()

    def is_connected_graph(self, graph):
        return nx.is_connected(graph)

    def remove_disconnected_communities(self, communities, G):
        connected_communities = []
        for community in communities:
            community_graph = G.subgraph(community)
            if self.is_connected_graph(community_graph):
                connected_communities.append(community)
        return connected_communities

    def networkx(self, filename):
        fin = open(filename, 'r')
        G = nx.Graph()
        for line in fin:
            data = line.split()
            if data[0] != '#':
                G.add_edge(int(data[0]), int(data[1]))
        return G

    def loadDataset(self, root, dataset):
        with open(f'{root}/{dataset}/{dataset}-1.90.ungraph.txt') as fh:
            edges = fh.read().strip().split('\n')
            edges = np.array([[int(i) for i in x.split()] for x in edges])
        with open(f'{root}/{dataset}/{dataset}-1.90.cmty.txt') as fh:
            comms = fh.read().strip().split('\n')
            comms = [[int(i) for i in x.split()] for x in comms]
        graph = Graph(edges)
        return graph, comms

    def loadEmbeddings(self):
        embeddings =  np.load(f"../embeddings/{self.args.dataset}_node_embeddings.npy", allow_pickle=True)
        embeddings_newid = {}

        for i in self.knowcomSeedGraph.neighbors.keys():
            embeddings_newid.update({int(i): embeddings[self.new_to_old_node_mapping[i]]})
        return embeddings_newid

    def init_expander(self):
        args = self.args
        device = self.device
        expander_model = Agent(args.hidden_size).to(device)  

        expander_optimizer = optim.Adam(expander_model.parameters(), lr=args.g_lr) 
        expander = Expander(args, self.knowcomSeedGraph, expander_model, expander_optimizer, device,
                            max_size=args.max_size)   
        return expander

    def clear_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path) 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 


    def detect(self):
        tic = time.time()

        for _ in range(self.args.epochs):
            self.train_expander()   

        taco = time.time()
        print(f'##  Training Time: {(taco - tic) // 60} min {(taco - tic) % 60}s')

        with open(f'./res/xu_mul/{self.args.dataset}_pred_com.txt', 'w') as file:
            pass 

        for s in self.seed:
            pred_com = [[s]]
            pred_com = self.expander.generateCommunity(pred_com)  
            pred_com = [x[:-1] if x[-1] == 'EOS' else x for x in pred_com]   
            oldID_pred_com = [self.new_to_old_node_mapping[node] for node in pred_com[0]]  
            with open(f'./res/xu_mul/{self.args.dataset}_pred_com.txt', 'a') as file:
                for node in oldID_pred_com:
                    file.write(f"{node} ")
                file.write("\n")

        with open(f'./res/xu_mul/{self.args.dataset}_seed.txt', 'w') as file:
            for i in self.oldSeed:
                file.write(f"{i} ")

        with open(f'./res/xu_mul/{self.args.dataset}_com_index.txt', 'w') as file:
            for j in self.com_index:
                file.write(f"{j} ")

        file_path_seed = f"res/xu_mul/{self.args.dataset}_seed.txt"
        seed_nodes = read_seed_nodes_from_file(file_path_seed)
        print("seeds_list：", seed_nodes)

        file_path_com_indices = f"res/xu_mul/{self.args.dataset}_com_index.txt"
        community_indices = read_community_indices(file_path_com_indices)
        print("seeds_com_list：", community_indices)

        file_path_xu = f"res/xu_mul/{self.args.dataset}_pred_com.txt"
        communities_xu = read_communities_from_file(file_path_xu)

        true_community_file_path = f"datasets/{self.args.dataset}/{self.args.dataset}-1.90.cmty.txt"
        communities_true = read_true_communities(true_community_file_path, community_indices)

   

        print("Results: Precision\t Recall\t\t\t\t\t Fscore\t\t\t\t\t Jeccard")
        num = len(communities_true)

 
        p_temp, r_temp, f_temp, j_temp = 0, 0, 0, 0
        for i in range(num):
            p, r, f, j = eval_scores(communities_xu[i], communities_true[i])
            p_temp += p
            r_temp += r
            f_temp += f
            j_temp += j
        p_avg = p_temp / num
        r_avg = r_temp / num
        f_avg = f_temp / num
        j_avg = j_temp / num
        print(p_avg, "\t", r_avg, "\t", f_avg, "\t", j_avg)

      
        toc = time.time()
        print(f'##  Overall Time: {(toc - tic) // 60} min {(toc - tic) % 60}s')


    def select_lists(self, matrix, n):
        num_lists = list(range(len(matrix)))     
        while len(num_lists) < n:             
            num_lists = num_lists + num_lists
        random_indices = np.random.choice(num_lists, size=n, replace=True)  
        selected_lists = [matrix[i] for i in random_indices] 
        return selected_lists


    def train_expander(self):
        seeds = []
        true_coms = self.select_lists(self.train_comms, self.args.g_batch_size)  
        for com in true_coms:
            seeds.append(random.choice(com)) 

  
        self.expander.trainReward(seeds, true_coms)  


        true_comms = random.choices(self.train_comms, k=self.args.g_batch_size) 
        true_comms = [self.knowcomSeedGraph.sample_expansion_from_community(x) for x in true_comms]  
        self.expander.train_from_sets(true_comms)   
