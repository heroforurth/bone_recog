
import os
import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from anomaly_model import ConvAutoencoder
import matplotlib.pyplot as plt

# Configuration
DATA_ROOT = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
MODEL_SAVE_PATH = "ae_bone.pth"
THRESHOLD_FILE = "ae_threshold.txt"
IMG_SIZE = 128 # Smaller size for AE is usually sufficient for structure
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Definition (Simplified for Reconstruction) ---
class BoneReconDataset(Dataset):
    def __init__(self, split="train"):
        self.image_paths = []
        
        # Valid categories - use BOTH fractured and not fractured as "Positive" class (Bone)
        categories = ["fractured", "not fractured"]
        
        split_dir = os.path.join(DATA_ROOT, split)
        
        if not os.path.exists(split_dir):
             print(f"Warning: {split} directory not found. Using train.")
             split_dir = os.path.join(DATA_ROOT, "train")

        for cat in categories:
            img_cat_dir = os.path.join(split_dir, cat)
            if not os.path.exists(img_cat_dir):
                continue
                
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                img_files = glob.glob(os.path.join(img_cat_dir, ext))
                self.image_paths.extend(img_files)
        
        print(f"[{split}] Found {len(self.image_paths)} bone images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Normalize 0-1
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) 
        
        # Return image as both input and target
        return torch.tensor(image), torch.tensor(image)

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    train_ds = BoneReconDataset(split="train")
    if len(train_ds) == 0:
        print("Error: No data found.")
        return

    # Split train into train/val
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} -> Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"Training finished. Best Val MSE: {best_val_loss:.6f}")
    
    # 4. Determine Threshold using Max MSE on Validation Set
    print("Calculating threshold...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    max_mse = 0
    all_mses = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            # Calculate MSE per image in batch
            # images: [B, 3, H, W], outputs: [B, 3, H, W]
            loss = (images - outputs) ** 2
            loss = loss.mean(dim=(1, 2, 3)) # [B]
            
            batch_max = loss.max().item()
            if batch_max > max_mse:
                max_mse = batch_max
            
            all_mses.extend(loss.cpu().numpy())

    # Set threshold slightly higher than max validation error (safety margin)
    # Or use 95th/99th percentile to avoid outliers
    threshold = np.percentile(all_mses, 99) 
    # threshold = max_mse * 1.1 # Alternative: 10% margin above max
    
    print(f"Max Val MSE: {max_mse:.6f}")
    print(f"99th Percentile MSE: {threshold:.6f}")
    
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))
    
    print(f"Saved threshold {threshold:.6f} to {THRESHOLD_FILE}")

if __name__ == "__main__":
    train()
