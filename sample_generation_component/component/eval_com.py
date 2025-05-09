from typing import Union, List, Set

def eval_scores(pred_comm: Union[List, Set],
                true_comm: Union[List, Set]) -> (float, float, float, float):

    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return round(p, 4), round(r, 4), round(f, 4), round(j, 4)


def read_seed_nodes_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip() 
            seed_nodes = list(map(int, content.split()))
            return seed_nodes
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []



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