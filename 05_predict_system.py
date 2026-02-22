
import os
import cv2
import torch
import numpy as np
import argparse
import segmentation_models_pytorch as smp
from anomaly_model import ConvAutoencoder
import matplotlib.pyplot as plt

# Configuration
AE_MODEL_PATH = "ae_bone.pth"
UNET_MODEL_PATH = "unet_best.pth"
THRESHOLD_FILE = "ae_threshold.txt"
AE_IMG_SIZE = 128
UNET_IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print(f"Using device: {DEVICE}")
    
    # Load Autoencoder
    print("Loading Autoencoder...")
    ae = ConvAutoencoder().to(DEVICE)
    if os.path.exists(AE_MODEL_PATH):
        ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: AE model {AE_MODEL_PATH} not found.")
        return None, None, None
    ae.eval()
    
    # Load Threshold
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            threshold = float(f.read().strip())
        print(f"Loaded Anomaly Threshold: {threshold}")
    else:
        print("Warning: Threshold file not found. Using default.")
        threshold = 0.05 # Default fallback
    
    # Load U-Net
    print("Loading U-Net...")
    unet = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    )
    if os.path.exists(UNET_MODEL_PATH):
        unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: U-Net model {UNET_MODEL_PATH} not found.")
    unet.to(DEVICE)
    unet.eval()
    
    return ae, unet, threshold

def preprocess_image(image_path, size):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (size, size))
    
    # Normalize
    input_tensor = resized / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)).astype(np.float32)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0) # Add batch dim
    
    return input_tensor, rgb_image

def predict(image_path):
    ae, unet, threshold = load_models()
    if ae is None:
        return

    # 1. Anomaly Detection (Is it a bone?)
    ae_input, original_ae = preprocess_image(image_path, AE_IMG_SIZE)
    if ae_input is None: return
    
    ae_input = ae_input.to(DEVICE)
    
    with torch.no_grad():
        reconstruction = ae(ae_input)
        mse = torch.mean((ae_input - reconstruction) ** 2).item()
    
    print(f"Reconstruction MSE: {mse:.6f} (Threshold: {threshold:.6f})")
    
    if mse > threshold:
        print(f"RESULT: ❌ NOT A BONE IMAGE (High Error: {mse:.6f})")
        is_bone = False
    else:
        print(f"RESULT: ✅ BONE DETECTED (Low Error: {mse:.6f})")
        is_bone = True

    # 2. Fracture Detection (if bone)
    if is_bone and unet:
        unet_input, original_unet = preprocess_image(image_path, UNET_IMG_SIZE)
        unet_input = unet_input.to(DEVICE)
        
        with torch.no_grad():
            output = unet(unet_input)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
        # Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_unet)
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f"Fracture Prediction")
        plt.axis('off')
        
        save_name = "prediction_result.png"
        plt.savefig(save_name)
        print(f"Prediction saved to {save_name}")
    else:
        print("Skipping fracture detection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()
    
    predict(args.image_path)
