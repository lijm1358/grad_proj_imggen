import pandas as pd
import os
import pickle
import argparse

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from typing import List
from fashion_clip.fashion_clip import FashionCLIP

BATCH_SIZE = 4 
WIDTH = 512
HEIGHT = 512
INFERENCE_STEP = 30

np.random.seed(42) # query 색상 랜덤 추가 시 seed 고정

items = pd.read_csv("./data/articles.csv")
cut_items = pd.read_csv("./data/additional_id_2024-04-28.csv")
# cut_items = pd.read_csv("./VBPR/new_item_data.csv")

items = items[items["article_id"].isin(cut_items["article_id"])]
items_desc = items["detail_desc"].values
items_id = items["article_id"].astype(str).values

gen_img_save_path = "./dataset_fashion/images/"
gen_latent_save_path = "./dataset_fashion/latents/"
gen_embed_save_path = "./dataset_fashion/embeddings/"


def dump_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def query_add_color() -> List[str]:
    new_item_desc = []
    for desc, color in zip(items_desc, items["colour_group_name"].values):
        if color == "Other":
            color = "Unknown"
        if len(color.split(" ")) == 2 and color.split(" ")[0] == "Other":
            color = color.split(" ")[1]
        if not isinstance(desc, str):
            new_item_desc.append("nan")
        else:
            if np.random.uniform() <= 0.7 and color != "Unknown":
                new_item_desc.append(f"{desc} {color} colored.")
            else:
                new_item_desc.append(f"{desc}")
                
    return new_item_desc

def main(args):
    number_gpu = args.number_gpu
    image_id = args.image_id if args.image_id is not None else number_gpu
    print(image_id)
    gen_log = f"./dataset_fashion/logs_{image_id}.txt"
    new_item_desc = query_add_color()
    
    fclip = FashionCLIP('fashion-clip')
    
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        custom_pipeline="clip_guided_stable_diffusion",
        torch_dtype = torch.float16,
        clip_model = clip_model,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(f"cuda:{number_gpu}")
    
    
    for i, prompt in enumerate(new_item_desc):
        try:
            print(f"[{i}/{len(new_item_desc)}] Generating images with id {items_id[i]} ({prompt})")
            
            filename = items_id[i]
            
            prompt += "Displayed against a white background, ensuring the entire item is visible and centrally placed. No one wears it."
            
            images = pipe(prompt, width=WIDTH, height=HEIGHT, num_inference_steps=INFERENCE_STEP)[0]
            
            os.makedirs(os.path.join(gen_img_save_path, filename), exist_ok=True)
            for j, img in enumerate(images):
                os.makedirs(os.path.join(gen_img_save_path, filename), exist_ok=True)
                img.save(os.path.join(gen_img_save_path, filename, f"{filename}_{image_id}.jpg"))
            
            with open(gen_log, "a") as file:
                file.write(f"{i} - {items_id[i]} : SUCCESS\n")
        except Exception as e:
            with open(gen_log, "a") as file:
                file.write(f"{i} - {items_id[i]} : ERROR ({e})\n")
            continue
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_gpu", dest="number_gpu", action="store", required=True)
    parser.add_argument("-i", "--image_id", dest="image_id", action="store", required=False)
    args = parser.parse_args()
    
    os.makedirs(gen_img_save_path, exist_ok=True)
    os.makedirs(gen_latent_save_path, exist_ok=True)
    os.makedirs(gen_embed_save_path, exist_ok=True)
    
    main(args)