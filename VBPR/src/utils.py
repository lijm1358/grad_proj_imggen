import os
import yaml
import pickle
import random
import numpy as np
import torch
from datetime import datetime

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_metric = float('-inf')

    def early_stop(self, metric):
        if metric > self.max_metric:
            self.max_metric = metric
            self.counter = 0
        elif metric < (self.max_metric - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_timestamp(date_format: str = '%d%H%M%S') -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)

def get_config(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def dump_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

def save_pt(data, path):
    with open(path, "wb") as file:
        torch.save(data, file)

def recall_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()
    return len(np.intersect1d(true, pred)) / len(true)

def ndcg_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()
    
    log2i = np.log2(np.arange(2, k + 2)) 
    dcg = np.sum(np.isin(pred, true) / log2i) # rel_i = 1
    idcg = np.sum((1 / log2i)[:min(len(true), k)]) 
    
    return dcg/idcg

def mk_dir(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    