import pandas as pd
import os
import pickle

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
INFERENCE_STEP = 25

np.random.seed(42) # query 색상 랜덤 추가 시 seed 고정

img_root = "./data/images/"
items = pd.read_csv("./data/articles_with_img.csv")
cut_items = pd.read_csv("./VBPR/new_item_data.csv")

items = items[items["article_id"].isin(cut_items["article_id"])]
items_desc = items["detail_desc"].values
items_id = items["article_id"].astype(str).values
img_path_list = [os.path.join("0" + id[:2], "0" + id + ".jpg") for id in items_id]

gen_img_save_path = "./dataset/images/"
gen_latent_save_path = "./dataset/latents/"
gen_embed_save_path = "./dataset/embeddings/"

gen_log = "./dataset/logs.txt"


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

def main():
    new_item_desc = query_add_color()
    
    fclip = FashionCLIP('fashion-clip')
    
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype = torch.float16,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    for i, prompt in enumerate(new_item_desc):
        try:
            print(f"[{i}/{len(new_item_desc)}] Generating images with id {items_id[i]} ({prompt})")
            
            filename = items_id[i]
            
            # latents = pipe([prompt] * BATCH_SIZE, width=WIDTH, height=HEIGHT, num_inference_steps=INFERENCE_STEP, output_type="latent")
            
            # images = pipe.vae.decode(latents.images / pipe.vae.config.scaling_factor, return_dict=False)[0].detach()
            # images = pipe.image_processor.postprocess(images, output_type="pil")
            # torch.cuda.empty_cache()
            
            images = pipe([prompt] * BATCH_SIZE, width=WIDTH, height=HEIGHT, num_inference_steps=INFERENCE_STEP).images
            latents = pipe.latents
            
            image_embeddings = fclip.encode_images(images, batch_size=4)
            
            os.makedirs(os.path.join(gen_img_save_path, filename), exist_ok=True)
            torch.save(latents, os.path.join(gen_latent_save_path, f"{filename}.pt"))
            for j, img in enumerate(images):
                img.save(os.path.join(gen_img_save_path, filename, f"{filename}_{j}.jpg"))
            dump_pickle(image_embeddings, os.path.join(gen_embed_save_path, f"{filename}.pkl"))
            
            with open(gen_log, "a") as file:
                file.write(f"{i} - {items_id[i]} : SUCCESS\n")
        except Exception as e:
            with open(gen_log, "a") as file:
                file.write(f"{i} - {items_id[i]} : ERROR ({e})\n")
            continue
    
        

if __name__ == '__main__':
    os.makedirs(gen_img_save_path, exist_ok=True)
    os.makedirs(gen_latent_save_path, exist_ok=True)
    os.makedirs(gen_embed_save_path, exist_ok=True)
    
    main()