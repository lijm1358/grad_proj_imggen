import torch
from tqdm import tqdm
from src.utils import recall_at_k

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

def eval(model, dataset, candidate_items_each_user, top_k, device):
    model.eval()
    metric = []
    
    with torch.no_grad():
         for user, target in tqdm(dataset):
            candidate_items = candidate_items_each_user[user].to(device)
            user = torch.tensor([user]).to(device)
            
            out, _ = model.cal_each(user, candidate_items)
            top_k_idx = out.argsort(descending=True)[:top_k]
            top_k_item = candidate_items[top_k_idx]

            metric.append(recall_at_k(target, top_k_item))
  
    return sum(metric)/len(metric)


