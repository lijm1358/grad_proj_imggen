from fashion_clip.fashion_clip import FashionCLIP
import os
import torch
import numpy as np
import random
from tqdm import tqdm

IMAGES_PATH = "./dataset_fashion/images"
EMBEDDINGS_PATH = "./dataset_fashion/embeddings"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main():
    fclip = FashionCLIP('fashion-clip')
    
    images_list = [os.path.join(IMAGES_PATH, image_id, image_id + f"_{i}.jpg") for image_id in os.listdir(IMAGES_PATH) for i in range(1, 4)]
    
    image_embeddings = fclip.encode_images(images_list, batch_size=128)
    
    print(f"size of image embeddings : {image_embeddings.shape}")
    torch.save(image_embeddings,os.path.join(EMBEDDINGS_PATH, "embeddings.pth"))
    
    image_embeddings = image_embeddings.reshape(-1, 3, 512)
    
    print(image_embeddings.shape)
    
    for i, image_id in enumerate(tqdm(os.listdir(IMAGES_PATH))):
        torch.save(image_embeddings[i], os.path.join(EMBEDDINGS_PATH, image_id + ".pth"))

if __name__ == '__main__':
    seed_everything()
    main()