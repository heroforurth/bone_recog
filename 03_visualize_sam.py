import os
import cv2
import numpy as np
import glob
import random
from tqdm import tqdm

# Configuration
DATA_ROOT = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
MASK_ROOT = "masks"
OUTPUT_DIR = "sam_visualizations"
SPLIT = "test"
CATEGORY = "fractured" # Focus on fractured for now as it's the target class

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_sam_results():
    # Paths
    img_dir = os.path.join(DATA_ROOT, SPLIT, CATEGORY)
    mask_dir = os.path.join(MASK_ROOT, SPLIT, CATEGORY)
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Error: Directories not found. Check {img_dir} and {mask_dir}")
        return

    # Get list of masks (since we want to verify what we generated)
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    
    if not mask_files:
        print("No masks found to visualize!")
        return
        
    print(f"Found {len(mask_files)} masks in {SPLIT}/{CATEGORY}")
    
    # Select random samples
    SAMPLE_SIZE = 20
    if len(mask_files) > SAMPLE_SIZE:
        sampled_masks = random.sample(mask_files, SAMPLE_SIZE)
    else:
        sampled_masks = mask_files
        
    print(f"Generating visualizations for {len(sampled_masks)} samples...")
    
    for mask_path in tqdm(sampled_masks):
        basename = os.path.basename(mask_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Try to find corresponding image (jpg or png)
        img_path_jpg = os.path.join(img_dir, name_no_ext + ".jpg")
        img_path_png = os.path.join(img_dir, name_no_ext + ".png")
        img_path_jpeg = os.path.join(img_dir, name_no_ext + ".jpeg")
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        elif os.path.exists(img_path_jpeg):
            img_path = img_path_jpeg
        else:
            print(f"Original image not found for mask: {basename}")
            continue
            
        # Load Image and Mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            continue
            
        # Resize mask to match image if needed (though they should match)
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Create Overlay
        # Mask is white (255) for fracture, black (0) for background
        # We want to make fracture red
        heatmap = np.zeros_like(image)
        heatmap[:, :, 2] = mask # Red channel
        
        # Alpha blending
        overlay = image.copy()
        alpha = 0.5
        mask_indices = mask > 0
        # Create heatmap overlay for the whole image
        heatmap_img = image.copy()
        heatmap_img[:, :, 2] = 255 # Make simple red channel
        
        # Weighted add to the whole image
        overlay_full = cv2.addWeighted(image, 1 - alpha, heatmap_img, alpha, 0)
        
        # Apply only to masked area
        overlay[mask_indices] = overlay_full[mask_indices]
        
        # Concatenate: Original | Mask | Overlay
        # Convert mask to 3 channel for concatenation
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(mask_bgr, "SAM Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        combined = np.hstack([image, mask_bgr, overlay])
        
        save_path = os.path.join(OUTPUT_DIR, f"vis_{name_no_ext}.jpg")
        cv2.imwrite(save_path, combined)
        
    print(f"Done! Check results in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    visualize_sam_results()
