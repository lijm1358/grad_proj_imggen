import os
import random
import dotenv
import wandb
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm

class HMDataset(Dataset):
    def __init__(self, df, user2idx, item2idx, is_train:bool=True) -> None:
        super().__init__()
        self.df = df
        self.is_train = is_train
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.n_user = len(self.user2idx)
        self.n_item = len(self.item2idx)
        # mapping id2idx
        self.df['article_id'] = self.df['article_id'].map(self.item2idx)
        self.df['customer_id'] = self.df['customer_id'].map(self.user2idx)
        
        # train 데이터인 경우에만 neg 아이템이 생성
        if is_train:
            self.df['neg'] = np.zeros(len(self.df), dtype=int)
            self._make_triples_data()
    
    def __getitem__(self, index):
        user = self.df.customer_id[index]
        pos = self.df.article_id[index]
        
        if self.is_train:
            neg = self.df.neg[index]
            return user, pos, neg
        
        return user, pos
    
    def _neg_sampling(self, pos_list):
        '''
        사용된 아이템 리스트(pos_list)에 없는 아이템 하나를 negative sample로 추출
        '''
        neg = np.random.randint(0,self.n_item,1) 
        while neg in pos_list:
            neg = np.random.randint(0,self.n_item,1) 
        return neg

    def _make_triples_data(self):
        for id in tqdm(range(self.n_user)):
            user_df = self.df[self.df.customer_id==id] # 유저 한 명 선택 
            pos_list = (user_df.article_id).tolist()   # 해당 유저가 사용한 아이템 모두 추출
            for i in range(len(user_df)): # 유저의 모든 구매 이력에 neg sample을 추가해줌
                idx = user_df.index[i] 
                self.df.at[idx, 'neg'] = self._neg_sampling(pos_list)
    
    def __len__(self):
        return len(self.df)

class VBPR(nn.Module):
    def __init__(self, n_user, n_item, K, D, img_embedding) -> None:
        super().__init__()
        self.feat_map= img_embedding.float() # user * 512
        self.n_user = n_user
        self.n_item = n_item
        self.K = K
        self.D = D
        self.F = self.feat_map.shape[1] 

        self.offset = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(self.n_user,1) # user*1
        self.item_bias = nn.Embedding(self.n_item,1) # item*1
        self.vis_bias = nn.Embedding(self.F,1)       # 512*1
        self.user_emb = nn.Embedding(self.n_user,self.K) # user*K
        self.item_emb = nn.Embedding(self.n_item,self.K) # item*K
        self.item_vis_emb = nn.Embedding(self.D, self.F) # D*K
        self.user_vis_emb = nn.Embedding(self.n_user, self.D) # user*D
    
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_bias.weight.data)
        nn.init.xavier_uniform_(self.vis_bias.weight.data)
        nn.init.xavier_uniform_(self.user_emb.weight.data)
        nn.init.xavier_uniform_(self.item_emb.weight.data)
        nn.init.xavier_uniform_(self.item_vis_emb.weight.data)
        nn.init.xavier_uniform_(self.user_vis_emb.weight.data)
    
    def cal_each(self, user, item):
        vis_term = (self.user_vis_emb(user)@(self.item_vis_emb.weight@(self.feat_map[item].T))).sum(dim=1) + (self.vis_bias.weight.T)@(self.feat_map[item].T)
        mf_term = self.offset + self.user_bias(user).T + self.item_bias(item).T + (self.user_emb(user)@self.item_emb(item).T).sum(dim=1).unsqueeze(dim=0)
        params = (self.offset, self.user_bias(user), self.item_bias(item), self.vis_bias.weight, self.user_emb(user), self.item_emb(item), self.item_vis_emb.weight, self.user_vis_emb(user))
        return (mf_term+vis_term).squeeze(), params
    
    def forward(self, user, pos, neg):
        xui, pos_params = self.cal_each(user,pos)
        xuj, neg_params = self.cal_each(user,neg)
        return (xui-xuj), pos_params, neg_params

class BPRLoss(nn.Module):
    def __init__(self, reg_theta, reg_beta, reg_e) -> None:
        super().__init__()
        self.reg_theta = reg_theta
        self.reg_beta = reg_beta
        self.reg_e = reg_e

    def _cal_l2(self, *tensors):
        total = 0
        for tensor in tensors:
            total += tensor.pow(2).sum()
        return 0.5 * total

    def _reg_term(self, pos_params, neg_params):
        alpha, beta_u, beta_pos, beta_prime_pos, gamma_u, gamma_pos, e_pos, theta_u = pos_params
        _, _, beta_neg, beta_prime_neg, _, gamma_neg, e_neg, _ = neg_params

        reg_out = self.reg_theta * self._cal_l2(alpha, beta_u, beta_pos, beta_neg, theta_u, gamma_u, gamma_pos, gamma_neg)
        reg_out += self.reg_beta * self._cal_l2(beta_prime_pos, beta_prime_neg)
        reg_out += self.reg_e * self._cal_l2(e_pos, e_neg)

        return reg_out

    def forward(self, diff, pos_params, neg_params):
        loss = -nn.functional.logsigmoid(diff).sum() # sigma(x_uij)
        loss += self._reg_term(pos_params, neg_params) # reg_term

        return loss
    
def train(model, optimizer, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for user, pos, neg in tqdm(dataloader):
        user = user.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        diff, pos_params, neg_params = model(user, pos, neg)
        loss = criterion(diff, pos_params, neg_params)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss/len(dataloader)

class EarlyStopper:
    def __init__(self, patience=4, min_delta=0.001):
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

def cal_auc_score(model, df, sample_user_ids, all_items, device):    
    grnd_truth = []
    scores = []
    
    for user_id in tqdm(sample_user_ids):
        user_df = df[df.customer_id==user_id]
        pos_ids = np.array(user_df.article_id)
        neg_ids = np.setdiff1d(all_items, pos_ids) # unobs item 집합
        np.random.shuffle(neg_ids)
        
        pos_ids = torch.tensor(pos_ids).to(device)
        neg_ids = torch.tensor(neg_ids[:len(pos_ids)]).to(device) # 새롭게 neg sampling
        user_ids = torch.tensor(np.full(len(pos_ids), user_id)).to(device)
        
        pos_pred, _ = model.cal_each(user_ids, pos_ids)
        neg_pred, _ = model.cal_each(user_ids, neg_ids)
        pred = pos_pred.tolist() + neg_pred.tolist()
        
        grnd_truth = np.zeros(2*len(pos_ids), dtype=np.int32)
        grnd_truth[:len(pos_ids)] = 1
        scores.append(roc_auc_score(grnd_truth, pred))
    
    return sum(scores)/len(scores)
    

def main():
    seed_everything()
    config_path = "./config/sweep2.yaml"
    config = get_config(config_path)
    print("--------------- Wandb SETTING ---------------")
    timestamp = get_timestamp()
    name = f"work-{timestamp}"

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="MMRec",
        name=name, 
        config=config)

    # hyper parameters
    K = wandb.config.K
    D = wandb.config.D
    reg_theta = wandb.config.reg_theta
    reg_beta = wandb.config.reg_beta
    reg_e = wandb.config.reg_e
    lr = wandb.config.lr
    epoch = wandb.config.epoch
    batch_size = wandb.config.batch_size
    
    # get img emb
    print("-------------LOAD IMAGE EMBEDDING-------------")
    img_emb = pd.read_csv("./data/img_emb.csv")
    img_emb = torch.tensor(img_emb.values)
    
    # load dataset
    print("-------------LOAD DATASET-------------")
    train_dataset = torch.load("./dataset/train_dataset.pt")
    # test_dataset = torch.load("./dataset/test_dataset.pt")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    
    # setting for training 
    n_user = train_dataset.n_user
    n_item = train_dataset.n_item
    sample_size = (n_user//100)*3
    sample_user_ids = np.random.choice(n_user, sample_size, replace=False) # 유저 샘플링
    all_items = np.arange(n_item, dtype=np.int32)

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    criterion = BPRLoss(reg_theta, reg_beta, reg_e).to(device)
    img_emb = img_emb.to(device)

    # train
    vbpr = VBPR(n_user, n_item, K, D, img_emb).to(device)
    optimizer = Adam(params = vbpr.parameters(), lr=lr)
    early_stopper = EarlyStopper()
    train_loss = []
    train_auc = []
    
    print("-------------TRAINING-------------")
    for i in range(epoch):
        train_loss.append(train(vbpr, optimizer, train_dataloader, criterion, device))
        train_auc.append(cal_auc_score(vbpr, train_dataset.df, sample_user_ids, all_items, device))
        
        print(f'EPOCH : {i} | AUC : {train_auc[-1]:.6} | LOSS : {train_loss[-1]:.6}')
        wandb.log({"train-auc":train_auc[-1] ,"train-loss":train_loss[-1], "epoch": i+1})
        
        if early_stopper.early_stop(train_auc[-1]):
            print("-------------EARLY STOPPING-------------")
            break
    
    torch.save(vbpr.state_dict(), "./model/"+name+".pt")
    wandb.save("./model/"+name+".pt")
    wandb.save(config_path)
    
    wandb.finish()
    

if __name__ == "__main__":
    main()