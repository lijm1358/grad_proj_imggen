import os 
import pandas as pd
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import random
import torch

DATASET_PATH = "./dataset_fashion"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
EMBEDDING_PATH = os.path.join(DATASET_PATH, "embeddings")
# CSV_PATH = "./data/articles_with_img.csv"
CSV_PATH = "./data/new_item_data.csv"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main():
    img_original_df = pd.read_csv(CSV_PATH)
    img_dir_list = os.listdir(IMAGE_PATH)
    
    try:
        assert len(img_original_df) == len(img_dir_list)
    except AssertionError:
        raise AssertionError(f"length of true image {len(img_original_df)} is not match with length of generated image {len(img_dir_list)}")
    
    try:
        for path in img_dir_list:
            target_path = os.listdir(os.path.join(IMAGE_PATH, path))
            img_count = len(target_path)
            assert img_count >= 3
            # assert img_count == 4
    except AssertionError:
        raise AssertionError(f"{target_path} has under 3 images")
    
    
    try:
        original_ids = img_original_df["article_id"].sort_values().to_numpy()
        gen_ids = sorted([int(id) for id in img_dir_list])

        assert np.all(np.equal(original_ids, gen_ids))
    except AssertionError:
        raise AssertionError(f"Original item ids and generated item ids does not match.")
    
    try:        
        target_ids = random.sample(img_dir_list, 10)
        
        print(f"Checking for 10 embeddings : {target_ids}")
        
        seed_everything(42)
        fclip = FashionCLIP('fashion-clip')
        
        for target_id in target_ids:
            original_emb = torch.load(os.path.join(EMBEDDING_PATH, target_id + ".pth"))
            original_emb = torch.from_numpy(original_emb)
            
            target_img_group = [os.path.join(IMAGE_PATH, target_id, target_id + f"_{i}.jpg") for i in range(1, 4)]
            
            target_embedding = fclip.encode_images(target_img_group, batch_size=3)
            target_embedding = torch.from_numpy(target_embedding)
            
            assert torch.allclose(original_emb, target_embedding, rtol=1e-4, atol=1e-5)
    except AssertionError:
        print(f"generated embedding and original embedding is not same.")
        
    

if __name__ == '__main__':
    main()