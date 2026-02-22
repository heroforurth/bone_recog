
import os
import cv2
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

# Configuration
DATA_ROOT = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
MASK_ROOT = "masks"
MODEL_PATH = "unet_best.pth"  # Or check checkpoints/ if used
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_IMAGE = "visualization_results.png"

# --- Dataset Definition (Same as training script) ---
class BoneDataset(Dataset):
    def __init__(self, split="val"):
        self.image_paths = []
        self.mask_paths = []
        self.split = split
        
        # Valid categories
        categories = ["fractured", "not fractured"]
        
        split_dir = os.path.join(DATA_ROOT, split)
        mask_split_dir = os.path.join(MASK_ROOT, split)
        
        # Use full dataset logic if split is empty (or random split logic if available)
        # But this script is for visualization, so we'll just check if files exist.
        if not os.path.exists(split_dir):
             print(f"Warning: {split} directory not found.")
             if split == "val":
                 print("Validation directory missing. Trying 'train' directory instead.")
                 split_dir = os.path.join(DATA_ROOT, "train")
                 mask_split_dir = os.path.join(MASK_ROOT, "train")

        for cat in categories:
            img_cat_dir = os.path.join(split_dir, cat)
            mask_cat_dir = os.path.join(mask_split_dir, cat)
            
            if not os.path.exists(img_cat_dir):
                continue
                
            # Find all images
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                img_files = glob.glob(os.path.join(img_cat_dir, ext))
                for img_p in img_files:
                    # Construct corresponding mask path
                    basename = os.path.basename(img_p)
                    name = os.path.splitext(basename)[0]
                    mask_name = name + ".png"
                    mask_p = os.path.join(mask_cat_dir, mask_name)
                    
                    if os.path.exists(mask_p):
                        self.image_paths.append(img_p)
                        self.mask_paths.append(mask_p)
        
        print(f"[{split}] Found {len(self.image_paths)} images with masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        original_image = image.copy()
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.where(mask > 127, 1.0, 0.0) # Binarize
        
        # Prepare for model (Normalize)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) 
        mask = mask.astype(np.float32)

        return torch.tensor(image), torch.tensor(mask), original_image, img_path

def visualize():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    val_ds = BoneDataset(split="val")
    if len(val_ds) == 0:
        print("Validation set empty, trying train set...")
        val_ds = BoneDataset(split="train")
    
    if len(val_ds) == 0:
        print("Error: No data found for visualization.")
        return

    # Select random samples
    indices = np.random.choice(len(val_ds), size=min(5, len(val_ds)), replace=False)
    
    # 2. Load Model
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    )
    
    # Check for model existence
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        if os.path.exists("checkpoints/best_model.pth"):
             model_path = "checkpoints/best_model.pth"
        elif os.path.exists("checkpoints/latest_model.pth"):
             model_path = "checkpoints/latest_model.pth"
        else:
             # Try listing files to see partial matches
             checkpoint_files = glob.glob("checkpoints/*.pth")
             if checkpoint_files:
                 model_path = checkpoint_files[0]
             else:
                 print(f"Error: Model file {MODEL_PATH} not found and no checkpoints found.")
                 return
    
    print(f"Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Predict and Plot
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Handle single sample case where axes is 1D
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    print("Generating visualization...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            input_tensor, mask, original_image, path = val_ds[idx]
            
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_batch)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
            
            # Plot Original
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original: {os.path.basename(path)}")
            axes[i, 0].axis('off')
            
            # Plot Ground Truth
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
            # Plot Prediction
            axes[i, 2].imshow(pred_mask_bin, cmap='gray')
            axes[i, 2].set_title(f"Predicted Mask (Conf: {pred_mask.mean():.2f})")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Visualization saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize()
