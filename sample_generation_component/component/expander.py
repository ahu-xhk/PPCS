from typing import Union, Optional, List, Set, Dict
import numpy as np
from scipy import sparse as sp
import torch
from torch import nn


from .env import ExpansionEnv
from .graph import Graph
from .gnn import GraphConv
from .agent import Agent


class Expander:

    def __init__(self,
                 args,
                 graph: Graph,
                 model: Agent,
                 optimizer,
                 device: Optional[torch.device] = None,
                 max_size: int = 25,

                 k: int = 3, 
                 alpha: float = 0.85,
                 gamma: float = 0.99,
                 ):
        self.graph = graph  
        self.model = model  
        self.optimizer = optimizer   
        self.n_nodes = self.graph.n_nodes  
        self.max_size = max_size 
        self.conv = GraphConv(graph, k, alpha)
        self.gamma = gamma
        self.args = args
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def updateGraphAndFeatAndConv(self, z_nodes, bs, new_nodes):
 
        oldId = self.args.new_to_old_node_mapping[new_nodes[0]]
        oldIdnodeKego = self.graph.parentGraph.k_ego([oldId], self.args.k_ego_subG)
        start_key = max(self.args.new_to_old_node_mapping.keys()) + 1
        newIDnode_nei = dict()
        for oldIdnode in oldIdnodeKego:
            if oldIdnode not in self.args.old_to_new_node_mapping:
        
                self.args.old_to_new_node_mapping[oldIdnode] = start_key
                self.args.new_to_old_node_mapping[start_key] = oldIdnode
                newIDnode_nei[start_key] = set()
                start_key += 1

        for newIdNode in newIDnode_nei.keys():
            oldIdnode = self.args.new_to_old_node_mapping[newIdNode]
            for oldIdnei_node in self.graph.parentGraph.neighbors[oldIdnode]:
                if oldIdnei_node in self.args.old_to_new_node_mapping:

                    newIDnode_nei[newIdNode].add(self.args.old_to_new_node_mapping[oldIdnode])
        if len(newIDnode_nei) != 0:

            self.graph.add_nodes_with_neighbors(newIDnode_nei)
            new_n_nodes = self.graph.n_nodes
            extended_matrix = sp.csc_matrix((new_n_nodes, bs), dtype=np.float32)
            extended_matrix[:self.n_nodes, :] = z_nodes
            z_nodes = extended_matrix
            self.n_nodes = new_n_nodes
            self.conv.updateGraph(self.graph)
        return z_nodes



    def _prepare_inputs(self,
                        valid_index: List[int],
                        trajectories: List[List[int]],
                        z_nodes: sp.csc_matrix,
                        z_seeds: sp.csc_matrix
                        ):

        vals_seed = []
        vals_node = []
        indptr = []
        offset = 0
        batch_candidates = []
        for i in valid_index:
            boundary_nodes = self.graph.outer_boundary(trajectories[i])     
            candidate_nodes = list(boundary_nodes)
            # assert len(candidate_nodes)
            involved_nodes = candidate_nodes + trajectories[i]  
            batch_candidates.append(candidate_nodes)  

            vals_seed.append(z_seeds.T[i, involved_nodes].todense())   
            vals_node.append(z_nodes.T[i, involved_nodes].todense())    
            indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))      
            offset += len(involved_nodes)

        vals_seed = np.array(np.concatenate(vals_seed, 1))[0]        
        vals_node = np.array(np.concatenate(vals_node, 1))[0]
        vals_seed = torch.from_numpy(vals_seed).to(self.device)
        vals_node = torch.from_numpy(vals_node).to(self.device)
        indptr = np.array(indptr)                

        return vals_seed, vals_node, indptr, batch_candidates


    def _sample_actions(self, batch_logits: List) -> (List, List, List):

        batch = []
        for logits in batch_logits:
            ps = torch.exp(logits) + 1e-8
     
            try:
                action = torch.multinomial(ps, 1).item()
            except Exception as e:
                print(f"Caught an exception: {e}")
                print(f"ps: {ps}")
                print(f"logits: {logits}")
            logp = logits[action]
            batch.append([action, logp])
        actions, logps = zip(*batch)
        actions = np.array(actions)

        return actions, logps



    def _sample_trajectories(self, env: ExpansionEnv, isTrain):

        bs = env.bs


        x_seeds, delta_x_nodes = env.reset()  

        z_seeds = self.conv(x_seeds) 

        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        new_nodes = []

        while not env.done:

            z_nodes += self.conv(delta_x_nodes)   

            if isTrain == False and len(new_nodes) != 0:
     
                z_nodes = self.updateGraphAndFeatAndConv(z_nodes, bs, new_nodes)
                env.updateGraph(self.graph)
                seeds = [env.data[i][0] for i in range(bs)]
                x_seeds = env.make_single_node_encoding(seeds)
                z_seeds = self.conv(x_seeds)

            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)  
            actions, logps = self._sample_actions(batch_logits)
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]  

            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)

        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)

 
        return env.trajectories, logps


    def generateCommunity(self, seeds: List[List[int]], max_size: Optional[int] = None):

        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, seeds, max_size, self.args.nodes_embeddings)
        self.model.eval()
        isTrain = False
        with torch.no_grad():
            episodes, _ = self._sample_trajectories(env, isTrain)         
        return episodes





    def sample_bs_trajectories(self, seeds: List[int], max_size: Optional[int] = None):

        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size, self.args.nodes_embeddings)
        isTrain = True
        return self._sample_trajectories(env, isTrain)

    def eval_scores(self, pred_comm: Union[List, Set],
                     true_comm: Union[List, Set]) -> (float, float, float, float):

        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm)
        r = len(intersect) / len(true_comm)
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    def tianchong(self, rewards, logps):
  
        max_len = max(len(sublist) for sublist in rewards)

        filled_rewards = np.array([np.pad(sublist, (0, max_len - len(sublist)), 'constant') for sublist in rewards])

        rows, cols = len(logps), len(logps[0])

        filled_rewards = np.pad(filled_rewards, ((0, rows - len(filled_rewards)), (0, cols - filled_rewards.shape[1])),
                                'constant')
        return filled_rewards

    def trainReward(self, seeds: List[int], true_coms):

        bs = len(seeds)
        self.model.train()
        self.optimizer.zero_grad()

        selected_nodes, logps = self.sample_bs_trajectories(seeds)


        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        rewards = []
        for index in range(len(selected_nodes)):
            com = selected_nodes[index]
            true_com = true_coms[index]
            r, gamma = [], 0.99
            temp_com = [com[0]]
            for node in com[1:]:
                if node != 'EOS':
                    _, _, pre_cost, _ = self.eval_scores(temp_com, true_com)
                    temp_com.append(node)
                    _, _, after_cost, _ = self.eval_scores(temp_com, true_com)
                    r.append(after_cost - pre_cost)
            reward = [np.sum(r[i] * (gamma ** np.array(range(i, len(r))))) for i in range(len(r))]
            rewards.append(reward)
        rewards = self.tianchong(rewards, logps)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        mask = torch.arange(rewards.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        policy_loss = -(rewards * logps * mask).sum()
        loss = policy_loss
        loss.backward()
        self.optimizer.step()




    def train_from_sets(self, episodes: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size, self.args.nodes_embeddings)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        while not env.done:
            k += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)
            logps = []
            actions = []
            for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    action = len(candidates)
                else:
                    sub_idx = [idx for idx, v in enumerate(candidates) if v in valid_candidates]
                    action = sub_idx[logits[sub_idx].argmax().item()]    
                actions.append(action)
                logps.append(logits[action])
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)
        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)

        lengths = torch.LongTensor([len(x) for x in env.trajectories]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        policy_loss = -(1 * logps * mask).sum() / n
        policy_loss.backward()
        self.optimizer.step()