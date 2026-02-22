
import os
import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DATA_ROOT = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
MASK_ROOT = "masks"
MODEL_SAVE_PATH = "unet_best.pth"
LOG_FILE = "training_log.txt"
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- Dataset Definition ---
class BoneDataset(Dataset):
    def __init__(self, split="train"):
        self.image_paths = []
        self.mask_paths = []
        
        # Valid categories
        categories = ["fractured", "not fractured"]
        
        split_dir = os.path.join(DATA_ROOT, split)
        mask_split_dir = os.path.join(MASK_ROOT, split)
        
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
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.where(mask > 127, 1.0, 0.0) # Binarize
        mask = np.expand_dims(mask, axis=0) # Channel First

        # Normalize Image
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) 
        mask = mask.astype(np.float32)

        return torch.tensor(image), torch.tensor(mask)

def main():
    # 1. Prepare Data
    train_ds = BoneDataset(split="train")
    val_ds = BoneDataset(split="val")
    
    # If val is empty, split train? User said dataset is split already.
    # If val is empty, just use random split from train?
    if len(val_ds) == 0 and len(train_ds) > 0:
        print("Validation set empty, splitting train set...")
        full_size = len(train_ds)
        train_size = int(0.8 * full_size)
        val_size = full_size - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    if len(train_ds) == 0:
        print("Error: No training data found! Check paths or run mask generation.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_ds)} samples, Validating on {len(val_ds)} samples.")

    # 2. Model Setup
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    )
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    
    # 3. Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    with open(LOG_FILE, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss\n")

    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch") as t:
            for images, masks in t:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
        
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model to {MODEL_SAVE_PATH}")

    # Plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig('training_loss_plot.png')
    print("Training finished.")

if __name__ == "__main__":
    main()
