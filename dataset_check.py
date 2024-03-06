import os 
import pandas as pd

DATASET_PATH = "./dataset_fashion"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
# CSV_PATH = "./data/articles_with_img.csv"
CSV_PATH = "./data/new_item_data.csv"

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
    
    

if __name__ == '__main__':
    main()