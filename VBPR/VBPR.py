import os
import random
import dotenv
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
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

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_timestamp(date_format: str = '%d%H%M%S') -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)

def main():
    seed_everything()
    K = 20
    D = 20
    reg_theta = 0.1
    reg_beta = 0.1
    reg_e = 0
    lr = 0.001
    epoch = 15
    
    print("--------------- Wandb Setting ---------------")
    timestamp = get_timestamp()
    name = f"work-{timestamp}"

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="MMRec",
        name=name, 
        config={
        "learning_rate": lr,
        "model": "vbpr",
        "dataset": "len cut 7",
        "epochs": epoch,
        "K" : K,
        "D" : D,
        "reg_theta" : reg_theta,
        "reg_beta" : reg_beta,
        "reg_e" : reg_e
        })
    
    # get img emb
    img_emb = pd.read_csv("./data/img_emb.csv")
    img_emb = torch.tensor(img_emb.values)
    
    # load dataset
    train_dataset = torch.load("./dataset/train_dataset.pt")
    test_dataset = torch.load("./dataset/test_dataset.pt")
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    
    # set hyper parameters
    n_user = train_dataset.n_user
    n_item = train_dataset.n_item

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    criterion = BPRLoss(reg_theta, reg_beta, reg_e).to(device)
    img_emb = img_emb.to(device)

    # training
    vbpr = VBPR(n_user, n_item, K, D, img_emb).to(device)
    optimizer = Adam(params = vbpr.parameters(), lr=lr)
    train_loss = []

    for i in range(epoch):
        train_loss.append(train(vbpr, optimizer, train_dataloader, criterion, device))
        print(f'EPOCH : {i} | LOSS : {train_loss[-1]:.10}')
        wandb.log({"train-loss":train_loss[-1], "epoch": i+1})
    
    torch.save(vbpr.state_dict(), "./model/"+name+".pt")
    wandb.save("./model/"+name+".pt")
    
    wandb.finish()

if __name__ == "__main__":
    main()