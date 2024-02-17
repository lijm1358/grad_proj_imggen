import torch
import numpy as np
from tqdm import tqdm
from src.utils import recall_at_k, ndcg_at_k

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

def eval(model, dataset, candidate_items_each_user, device):
    model.eval()
    metrics = {'R10':[], 'R20':[], 'R40':[], 'N10':[], 'N20':[], 'N40':[]}

    with torch.no_grad():
         for user, target in tqdm(dataset):
            candidate_items = candidate_items_each_user[user].to(device)
            user = torch.tensor([user]).to(device)
            target = torch.tensor([target]).to(device)
            
            out, _ = model.cal_each(user, candidate_items)
            sorted_idx = out.argsort(descending=True)
            sorted_item = candidate_items[sorted_idx]

            for k in [10, 20, 40]:
                metrics['R' + str(k)].append(recall_at_k(k, target, sorted_item))
                metrics['N' + str(k)].append(ndcg_at_k(k, target, sorted_item))
              
    for k in [10, 20, 40]:
        metrics['R' + str(k)] = round(np.asarray(metrics['R' + str(k)]).mean(), 5)   
        metrics['N' + str(k)] = round(np.asarray(metrics['N' + str(k)]).mean(), 5)
        
    return metrics


