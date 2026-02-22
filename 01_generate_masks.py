
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from ultralytics import SAM

# Configuration
# Pointing to the nested directory structure based on user feedback
DATA_ROOT = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
OUTPUT_ROOT = "masks"

def process_split(split_name):
    # e.g. split_name = "train", "test", "val"
    split_dir = os.path.join(DATA_ROOT, split_name)
    if not os.path.exists(split_dir):
        print(f"Warning: Split '{split_name}' not found at {split_dir}")
        return

    # Categories: "fractured" and "not fractured"
    categories = ["fractured", "not fractured"]
    
    for cat in categories:
        cat_dir = os.path.join(split_dir, cat)
        if not os.path.exists(cat_dir):
            # Check for different casing if needed, but based on ls it's lowercase or 'not fractured'
            # ls output showed: fractured/  'not fractured'/
            # So "not fractured" folder exists.
            print(f"Warning: Category '{cat}' not found in {split_name}")
            continue
            
        # Create output directory
        out_cat_dir = os.path.join(OUTPUT_ROOT, split_name, cat)
        os.makedirs(out_cat_dir, exist_ok=True)
        
        # Find images
        # Extensions might be jpg/png/jpeg
        image_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
            image_files.extend(glob.glob(os.path.join(cat_dir, ext)))
        
        # --- FAST TRACK: Limit number of images ---
        # User needs results tonight. 
        # 4606 images took ~4.5 hours. 
        # Limiting to 300 per category allows finishing in ~30 mins.
        LIMIT = 300
        if len(image_files) > LIMIT:
            print(f"Limiting {split_name}/{cat} from {len(image_files)} to {LIMIT} images for speed.")
            # Random shuffle to get variety
            np.random.shuffle(image_files)
            image_files = image_files[:LIMIT]
            
        print(f"Processing {len(image_files)} images in {split_name}/{cat}...")
        
        for img_path in tqdm(image_files):
            filename = os.path.basename(img_path)
            # Save as PNG to avoid compression artifacts in masks
            save_name = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(out_cat_dir, save_name)
            
            if os.path.exists(save_path):
                continue
                
            try:
                # Predict with SAM 2
                results = model(img_path, verbose=False)
                
                if results and len(results) > 0 and results[0].masks is not None:
                    # Combine all masks detected
                    masks = results[0].masks.data.cpu().numpy()
                    final_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                    cv2.imwrite(save_path, final_mask)
                else:
                    # Empty mask
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    cv2.imwrite(save_path, np.zeros((h, w), dtype=np.uint8))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Initialize Model (Global)
print("Loading SAM 2 (Ultralytics)...")
try:
    model = SAM("sam2.1_b.pt")
except Exception as e:
    print(f"Error loading SAM 2: {e}")
    exit(1)

def main():
    splits = ["train", "val", "test"]
    for split in splits:
        process_split(split)
    print("Mask generation finished.")

if __name__ == "__main__":
    main()
