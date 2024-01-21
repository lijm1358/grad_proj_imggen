import numpy as np
import torch
from tqdm import tqdm
from src.utils import recall_at_k
from torch import sigmoid

def train(model, optimizer, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for user, pos, neg in tqdm(dataloader):
        user = user.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        
        if criterion._get_name() != "BCELoss":
            diff, pos_params, neg_params = model(user, pos, neg)
            loss = criterion(diff, pos_params, neg_params)
        else:
            out = sigmoid(model(user, pos))
            target = torch.ones_like(neg).float()
            loss = criterion(out, target)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss/len(dataloader)

def eval(model, dataloader, all_item, pos_items_each_user, device, sample_size):
    model.eval()
    metric = []
    pred_list = {}
    sample_size = sample_size+1
    
    with torch.no_grad():
        for users, targets in tqdm(dataloader):
            idx = sample_size
            candidate_items = torch.tensor([], dtype=int).to(device)
            user = torch.tensor([], dtype=int).to(device)

            for i in range(users.shape[0]):
                u = users[i].item()
                t = targets[i].item()
                items = torch.tensor(np.append(np.random.choice(np.setdiff1d(all_item, pos_items_each_user[u]), sample_size-1), t)).to(device)
                u_ids = torch.tensor(np.full(sample_size, u)).to(device)
                candidate_items = torch.cat([candidate_items, items], dim=0)
                user = torch.cat([user, u_ids], dim=0)
            
            if model._get_name() == "VBPR":
                out, _ = model.cal_each(user, candidate_items)
            else:
                out = sigmoid(model(user, candidate_items))
            
            for target in targets:
                if idx-sample_size<0:
                    print("ERROR: idx is larger than total length")
                    break
                user_res = out[idx-sample_size:idx]
                user_cadidate = candidate_items[idx-sample_size:idx]
                top_k_idx = user_res.argsort(descending=True)[:20]
                pred_list[target.item()] = user_cadidate[top_k_idx]
                metric.append(recall_at_k(target.item(), user_cadidate[top_k_idx]))
                idx += sample_size
  
    return sum(metric)/len(metric), pred_list


'''
 for user, target in tqdm(dataloader):
            candidate_items = torch.tensor(np.append(np.random.choice(np.setdiff1d(all_item, pos_items_each_user[user]), sample_size), target)).to(device)
            user = torch.tensor([user]).to(device)
            
            out, _ = model.cal_each(user, candidate_items)
            top_10_idx = out.argsort(descending=True)[:10]
             
            metric.append(recall_at_k(target, candidate_items[top_10_idx]))
'''


