import numpy as np
import torch
import random

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def getFileInfo(type):
    seed = []
    filename = f'datasets/{type}'
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = list(map(int, line))
            seed.append(line)
    return seed

def getseedsAndtruecom(args, dataset):
    dic = {'amazon': 0, 'dblp': 1, 'lj': 2, 'youtube': 3, 'twitter': 4, 'facebook': 5}
    seeds = getFileInfo(f"seed12")[dic[dataset]]
    com_indexs = getFileInfo(f"com_index12")[dic[dataset]]
    return seeds, com_indexs
