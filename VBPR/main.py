import os
import dotenv
import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
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

    ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project="MMRec", name=name, config=config)

    ############ SET HYPER PARAMS #############
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
    emb_norm = wandb.config.emb_norm
    
    ############# LOAD DATASET #############
    print("-------------LOAD IMAGE EMBEDDING-------------")
    img_emb = torch.tensor(load_pickle("./data/img_emb_small.pkl"))
    print("-------------LOAD DATASET-------------")
    train_dataset = torch.load("./dataset/train_dataset_small.pt")
    test_dataset = torch.load("./dataset/test_dataset_small.pt")
    candidate_items_each_user = load_pickle("./data/candidate_items_each_user_small.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

    ############# SETTING FOR TRAIN #############
    n_user = len(test_dataset)
    n_item = img_emb.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    img_emb = img_emb.to(device)

    if model_name == "MF":
        model = BPRMF(n_user, n_item, K).to(device)
        criterion = BPRLoss(visual=False, reg_theta = reg_theta)
    else:
        model = VBPR(n_user, n_item, K, D, img_emb, emb_norm).to(device)
        criterion = BPRLoss(visual=True, reg_theta=reg_theta, reg_beta=reg_beta, reg_e=reg_e).to(device)

    optimizer = Adam(params = model.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch : 0.85**epoch)
    
    ############# TRAIN AND EVAL #############
    print("-------------TRAINING-------------")
    for i in range(epoch):
        train_loss = train(model, optimizer, scheduler, train_dataloader, criterion, device)
        print(f'EPOCH : {i+1} | LOSS : {train_loss} | lr : {optimizer.param_groups[0]["lr"]}')
        wandb.log({"loss":train_loss, "epoch": i+1, "lr": optimizer.param_groups[0]["lr"]})
        
        if i%3 == 0:
            metrics = eval(model, test_dataset, candidate_items_each_user, device)
            print(f'R10 : {metrics["R10"]} | R20 : {metrics["R20"]} | R40 : {metrics["R40"]} | N10 : {metrics["N10"]} | N20 : {metrics["N20"]} | N40 : {metrics["N40"]}')
            wandb.log(metrics)

    ############# WANDB FINISH & SAVING FILES #############
    wandb.save(config_path)
    wandb.finish()

if __name__ == "__main__":
    main()