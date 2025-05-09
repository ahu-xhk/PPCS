import argparse
import os
import shutil
from typing import List, Union, Set
import numpy as np
import torch
import time
import collections

import utils
import data
import model

def read_community_indices(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()  
            community_indices = list(map(int, content.split()))
            return community_indices
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []

def read_true_communities(file_path, indices):
    try:
        with open(file_path, 'r') as file:
            communities = [list(map(int, line.strip().split())) for line in file]
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

    selected_communities = [communities[i] for i in indices if i < len(communities)]
    return selected_communities

def eval_scores(pred_comm: Union[List, Set],
                true_comm: Union[List, Set]) -> (float, float, float, float):
 
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

def read_communities_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            communities = []
            for line in file:
 
                nodes = list(map(int, line.strip().split()))
                communities.append(nodes)
            return communities
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []

def read_train_communities_from_file(file_path):
    communities = []
    with open(file_path, 'r') as file:
        for line in file:
  
            line = line.strip()
            if line:  
                community = list(map(int, line.split(',')))
                communities.append(community)
    return communities

def get_communities_embeddings(communities: List[List[int]], all_nodes_embeddings: np.ndarray):
    community_embedding = []
    for com in communities:
        temp_emb = np.zeros_like(all_nodes_embeddings[0]) 
        for node in com:
            temp_emb += all_nodes_embeddings[node] 
        community_embedding.append(temp_emb)
    return community_embedding

def get_graph_neighbors(edges):
    neighbors = collections.defaultdict(set)
    for u, v in edges:
        if u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return neighbors

def outer_boundary(node_list, neighbors):
    boundary = set()
    for u in node_list:
        boundary |= neighbors[u] 
    boundary.difference_update(node_list)
    return boundary

def find_k_ego_com(node_list, k, neighbors):
    nodes = node_list.copy()
    ego_nodes = set(nodes)
    current_boundary = set(nodes)
    for _ in range(k):
        current_boundary = outer_boundary(current_boundary, neighbors) - ego_nodes
        ego_nodes |= current_boundary  
    return ego_nodes


if __name__ == '__main__':

    print('=' * 20)
    print('## Starting Time:', utils.get_cur_time(), flush=True)

    parser = argparse.ArgumentParser()

    # Datasets preparing
    parser.add_argument("--dataset", type=str, default='lj')
    parser.add_argument("--seeds", type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--device", type=str, default="cpu")

    # training related
    parser.add_argument("--prompt_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--generate_k", type=int, default=2)
    parser.add_argument("--final_predict_k", type=int, default=0)
    parser.add_argument("--similar_m", type=int, default=20)

    # GNN related
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Prompt related
    parser.add_argument("--threshold", type=float, default=0.05)

    args = parser.parse_args()

    print(args)

    with open(f'../data/{args.dataset}/{args.dataset}-1.90.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        edges = np.array([[int(i) for i in x.split()] for x in edges])
    neighbors = get_graph_neighbors(edges)

    _, _, _, graph_data, nx_graph, _ = data.prepare_data(args.dataset)

    all_nodes_embeddings = np.load(f"../../embeddings/{args.dataset}_node_embeddings.npy", allow_pickle=True)

    file_path_xu = f"../../expander_component_multiple_coms/res/xu_mul/{args.dataset}_pred_com.txt"
    communities_xu = read_communities_from_file(file_path_xu)
    predicted_coms_embeddings = get_communities_embeddings(communities_xu, all_nodes_embeddings)


    filename = f"../../expander_component_multiple_coms/datasets/train_coms/{args.dataset}_train_coms.txt"
    train_coms = read_train_communities_from_file(filename)
    train_coms_embeddings = get_communities_embeddings(train_coms, all_nodes_embeddings)


    predicted_coms_embeddings = np.array(predicted_coms_embeddings)
    train_coms_embeddings = np.array(train_coms_embeddings)
    m = args.similar_m 
    similar_train_coms = []
    for i in range(len(communities_xu)):
        single_com_embedding = predicted_coms_embeddings[i, :]
        distance = np.sqrt(np.sum(np.asarray(single_com_embedding - train_coms_embeddings) ** 2, axis=1))
        sort_dic = list(np.argsort(distance))
        length = 0
        for idx in sort_dic:
            if length >= m:
                break
            single_similar_com = train_coms[idx]
            similar_train_coms.append(single_similar_com)
            length += 1

    print("similar_train_coms: ", len(similar_train_coms))

    device = torch.device(args.device)
    utils.set_seed(args.seeds[0])


    prompt_model = model.PromptLinearNet(args.hidden_dim, threshold=args.threshold).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(prompt_model.parameters(), lr=args.lr, weight_decay=0.00001)

    num_prompt_param = sum(p.numel() for p in prompt_model.parameters())
    print(f"[Parameters] Number of parameters in Prompt {num_prompt_param}")


    num_iteration = len(similar_train_coms) // m
    if len(similar_train_coms) % m != 0:
        num_iteration += 1
    assert num_iteration == len(communities_xu)


    candidate_comms = []

    file_path_com_indices = f"../../expander_component_multiple_coms/res/xu_mul/{args.dataset}_com_index.txt"
    community_indices = read_community_indices(file_path_com_indices)

    true_community_file_path = f"../../expander_component_multiple_coms/datasets/{args.dataset}/{args.dataset}-1.90.cmty.txt"
    communities_true = read_true_communities(true_community_file_path, community_indices)

    with open(f'../res/{args.dataset}_pred_com.txt', 'w') as file:
        pass

    all_time = 0
    for i in range(num_iteration):
        st_time = time.time()
        start_idx = i * m
        end_idx = (i + 1) * m

        current_train_coms = similar_train_coms[start_idx:end_idx]

        current_train_coms.append(communities_xu[i])

        current_train_coms_embeddings = get_communities_embeddings(current_train_coms, all_nodes_embeddings)

        for epoch in range(args.prompt_epoch):
            prompt_model.train()
            optimizer.zero_grad()

            all_central_nodes, all_ego_nodes, all_labels = torch.FloatTensor().to(device), torch.FloatTensor().to(
                device), torch.FloatTensor().to(device)


            for community in current_train_coms:
                central_node, k_ego, label = utils.generate_prompt_tuning_data(community, graph_data, nx_graph,
                                                                               args.generate_k)
                central_node_emb = all_nodes_embeddings[central_node, :]
                ego_node_emb = all_nodes_embeddings[k_ego, :]

                central_node_emb_tensor = torch.from_numpy(central_node_emb)
                ego_node_emb_tensor = torch.from_numpy(ego_node_emb)

                all_labels = torch.cat((all_labels, label.to(device)), dim=0)
                all_central_nodes = torch.cat((all_central_nodes, central_node_emb_tensor), dim=0)
                all_ego_nodes = torch.cat((all_ego_nodes, ego_node_emb_tensor), dim=0)

            pred_logits = prompt_model(all_ego_nodes, all_central_nodes)

            pt_loss = loss_fn(pred_logits, all_labels)
            pt_loss.backward()
            optimizer.step()

        prompt_model.eval()


        node = communities_xu[i][0]


        node_k_ego = list(find_k_ego_com(communities_xu[i], args.final_predict_k, neighbors))

        final_pos = prompt_model.make_prediction(torch.from_numpy(all_nodes_embeddings[node_k_ego, :]),
                                                 torch.from_numpy(all_nodes_embeddings[[node] * len(node_k_ego), :]))

        if len(final_pos) > 0:
            candidate = [node_k_ego[idx] for idx in final_pos]
            if node not in candidate:
                candidate = [node] + candidate
            candidate_comms.append(candidate)
        else:
            candidate = communities_xu[i]
            if node not in candidate:
                candidate = [node] + candidate
            candidate_comms.append(candidate)

        now_time = time.time() - st_time
        all_time += now_time
        print(f"[No.{i}] Finish community {node} prediction, Cost Time {now_time:.5}s!")
        with open(f'../res/{args.dataset}_pred_com.txt', 'a') as file:
            for n in candidate_comms[i]:
                file.write(f"{n} ")
            file.write("\n")

    avg_time = all_time / num_iteration
    print(f"Average cost time {avg_time:.5}s!")

    file_path_com_indices = f"../../expander_component_multiple_coms/res/xu_mul/{args.dataset}_com_index.txt"
    community_indices = read_community_indices(file_path_com_indices)

    file_path_xu = f"../res/{args.dataset}_pred_com.txt"
    communities_xu = read_communities_from_file(file_path_xu)

    true_community_file_path = f"../../expander_component_multiple_coms/datasets/{args.dataset}/{args.dataset}-1.90.cmty.txt"
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


