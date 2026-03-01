import streamlit as st
import os
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from anomaly_model import ConvAutoencoder

# --- การตั้งค่าไฟล์โมเดล (ตรวจสอบชื่อไฟล์ให้ตรงกับที่คุณมี) ---
SAM_MODEL_PATH = "sam2.1_b.pt"
UNET_MODEL_PATH = "unet_best.pth"
AE_MODEL_PATH = "ae_bone.pth"
THRESHOLD_FILE = "ae_threshold.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Bone Recognition System", layout="wide")

# --- ฟังก์ชันตรวจสอบไฟล์โมเดล ---
def check_models_exist():
    missing_files = []
    # รายการไฟล์ที่ระบบต้องใช้
    for file in [SAM_MODEL_PATH, UNET_MODEL_PATH, AE_MODEL_PATH]:
        if not os.path.exists(file):
            missing_files.append(file)
    return missing_files

# --- ส่วนโหลดโมเดล ---
@st.cache_resource
def load_trained_models():
    # โหลด Autoencoder (สำหรับการเช็คว่าเป็นรูปกระดูกไหม)
    ae = ConvAutoencoder().to(DEVICE)
    if os.path.exists(AE_MODEL_PATH):
        ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae.eval()

    # โหลด U-Net (สำหรับการหาจุดแตกหัก)
    unet = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    )
    if os.path.exists(UNET_MODEL_PATH):
        unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
    unet.to(DEVICE)
    unet.eval()

    return ae, unet

# --- หน้าจอหลักของ Web App ---
st.title("🦴 ระบบวิเคราะห์กระดูกและตรวจจับรอยร้าว")

# ตรวจสอบไฟล์ก่อนเริ่ม
missing = check_models_exist()
if missing:
    st.error(f"❌ ไม่พบไฟล์โมเดลในโฟลเดอร์: {', '.join(missing)}")
    st.info("กรุณานำไฟล์โมเดล (.pt หรือ .pth) มาวางไว้ที่เดียวกับไฟล์ app.py")
    st.stop() 

# โหลดโมเดลเข้า Memory
ae_model, unet_model = load_trained_models()

uploaded_file = st.file_uploader("ลากและวางรูป X-ray ที่นี่ (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # อ่านรูปภาพ
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="รูป X-ray ต้นฉบับ", width=500)
    
    if st.button("เริ่มการวิเคราะห์ (Predict)"):
        with st.spinner("ระบบกำลังประมวลผล..."):
            # ดึง Logic การทำนายมาจาก 05_predict_system.py
            # 1. จัดการขนาดรูปสำหรับ AE (128x128)
            img_np = np.array(img)
            ae_input = cv2.resize(img_np, (128, 128)) / 255.0
            ae_tensor = torch.tensor(ae_input).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            
            # 2. ตรวจสอบความผิดปกติ (MSE)
            with torch.no_grad():
                recon = ae_model(ae_tensor)
                mse = torch.mean((ae_tensor - recon) ** 2).item()
            
            # โหลด Threshold
            threshold = 0.05
            if os.path.exists(THRESHOLD_FILE):
                with open(THRESHOLD_FILE, "r") as f:
                    threshold = float(f.read().strip())

            if mse > threshold:
                st.warning(f"⚠️ ผลลัพธ์: ตรวจพบค่า Error สูง ({mse:.4f}) รูปนี้อาจไม่ใช่กระดูกหรือมีความผิดปกติสูงมาก")
            else:
                st.success(f"✅ ผลลัพธ์: ตรวจพบกระดูกปกติ (MSE: {mse:.4f})")
                
                # 3. Segmentation (U-Net) เพื่อหารอยร้าว
                unet_input = cv2.resize(img_np, (512, 512)) / 255.0
                unet_tensor = torch.tensor(unet_input).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                
                with torch.no_grad():
                    mask = torch.sigmoid(unet_model(unet_tensor)).squeeze().cpu().numpy()
                
                # แสดงผล Mask
                st.subheader("ผลการวิเคราะห์รอยร้าว (Fracture Mask)")
                st.image(mask, caption="พื้นที่ที่คาดว่าจะมีรอยร้าว", use_container_width=True, clamp=True)