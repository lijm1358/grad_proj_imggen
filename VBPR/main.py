import os
import dotenv
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from src.utils import seed_everything, get_config, get_timestamp, load_pickle, dump_pickle, mk_dir
from src.model import VBPR, BPRLoss, MF
from src.dataset import HMTestDataset, HMTrainDataset
from src.train import train, eval

def main():
    ############# SETTING #############
    seed_everything()
    mk_dir("./model")
    mk_dir("./data/res")
    config_path = "./config/sweep.yaml"
    config = get_config(config_path)
    timestamp = get_timestamp()
    name = f"work-{timestamp}"

    ############# WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="MMRec",
        name=name, 
        config=config)

    ############# SET HYPER PARAMS #############
    K = wandb.config.K
    D = wandb.config.D
    reg_theta = wandb.config.reg_theta
    reg_beta = wandb.config.reg_beta
    reg_e = wandb.config.reg_e
    lr = wandb.config.lr
    epoch = wandb.config.epoch
    batch_size = wandb.config.batch_size   
    sample_size = wandb.config.sample_size
    model_name = wandb.config.model
    
    ############# LOAD DATASET #############
    print("-------------LOAD IMAGE EMBEDDING-------------")
    img_emb = pd.read_csv("./data/img_emb_new.csv")
    img_emb = torch.tensor(img_emb.values)
    print("-------------LOAD DATASET-------------")
    train_dataset = torch.load("./dataset/train_dataset_v1.pt")
    test_dataset = torch.load("./dataset/test_dataset_v1.pt")
    pos_items_each_user = load_pickle("./data/pos_items_each_user_small.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=128) # gpu 메모리에 따라 배치 사이즈 설정

    ############# SETTING FOR TRAIN #############
    n_user = len(test_dataset)
    n_item = img_emb.shape[0]
    all_items = np.arange(n_item, dtype=np.int32)

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    img_emb = img_emb.to(device)

    if model_name == "MF":
        model = MF(n_user, n_item, K).to(device)
        criterion = BCELoss().to(device)
    else:
        model = VBPR(n_user, n_item, K, D, img_emb).to(device)
        criterion = BPRLoss(reg_theta, reg_beta, reg_e).to(device)

    optimizer = Adam(params = model.parameters(), lr=lr)
    train_loss = []
    metrics = []
    
    ############# TRAIN AND EVAL #############
    print("-------------TRAINING-------------")
    for i in range(epoch):
        train_loss.append(train(model, optimizer, train_dataloader, criterion, device))
        if i%3 == 0:
            metric, pred_list = eval(model, test_dataloader, all_items, pos_items_each_user, device, sample_size)
            metrics.append(metric)
            dump_pickle(pred_list, f"./data/res/{timestamp}_{i}.pkl")
            wandb.save(f"./data/res/{timestamp}_{i}.pkl")
            print(f'EPOCH : {i} | Recall : {metrics[-1]:.6} | LOSS : {train_loss[-1]:.6}')
            wandb.log({"recall":metrics[-1] ,"loss":train_loss[-1], "epoch": i+1})
        else: 
            print(f'EPOCH : {i} | LOSS : {train_loss[-1]:.6}')
            wandb.log({"loss":train_loss[-1], "epoch": i+1})

    ############# WANDB FINISH & SAVING FILES #############
    torch.save(model.state_dict(), "./model/"+name+".pt")
    wandb.save("./model/"+name+".pt")
    wandb.save(config_path)
    wandb.finish()
    

if __name__ == "__main__":
    main()