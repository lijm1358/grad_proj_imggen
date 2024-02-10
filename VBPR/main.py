import os
import dotenv
import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.utils import seed_everything, get_config, get_timestamp, load_pickle, mk_dir
from src.model import VBPR, BPRLoss, BPRMF
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
    vis_weight = wandb.config.vis_weight
    top_k = wandb.config.top_k
    
    ############# LOAD DATASET #############
    print("-------------LOAD IMAGE EMBEDDING-------------")
    img_emb = torch.tensor(load_pickle("./data/img_emb_small.pkl"))
    print("-------------LOAD DATASET-------------")
    train_dataset = torch.load("./dataset/train_dataset_small.pt")
    test_dataset = torch.load("./dataset/test_dataset_small.pt")
    candidate_items_each_user = load_pickle("./data/candidate_items_each_user_small.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    ############# SETTING FOR TRAIN #############
    n_user = len(test_dataset)
    n_item = img_emb.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    img_emb = img_emb.to(device)

    if model_name == "MF":
        model = BPRMF(n_user, n_item, K).to(device)
        criterion = BPRLoss(visual=False, reg_theta = reg_theta)
    else:
        model = VBPR(n_user, n_item, K, D, img_emb, vis_weight).to(device)
        criterion = BPRLoss(visual=True, reg_theta=reg_theta, reg_beta=reg_beta, reg_e=reg_e).to(device)

    optimizer = Adam(params = model.parameters(), lr=lr)
    train_loss = []
    metrics = []
    
    ############# TRAIN AND EVAL #############
    print("-------------TRAINING-------------")
    for i in range(epoch):
        train_loss.append(train(model, optimizer, train_dataloader, criterion, device))
        if i%3 == 0:
            metrics.append(eval(model, test_dataset, candidate_items_each_user, top_k, device))
            print(f'EPOCH : {i+1} | Recall : {metrics[-1]:.6} | LOSS : {train_loss[-1]:.6}')
            wandb.log({"recall":metrics[-1] ,"loss":train_loss[-1], "epoch": i+1})
        else: 
            print(f'EPOCH : {i+1} | LOSS : {train_loss[-1]:.6}')
            wandb.log({"loss":train_loss[-1], "epoch": i+1})

    ############# WANDB FINISH & SAVING FILES #############
    wandb.save(config_path)
    wandb.finish()
    

if __name__ == "__main__":
    main()