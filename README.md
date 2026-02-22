# Bone Recognition & Fracture Detection

This repository contains a complete pipeline for bone segmentation and fracture detection using X-ray images. The system leverages state-of-the-art deep learning models, including SAM 2 (Segment Anything Model) for automated mask generation, U-Net for bone segmentation, and an Autoencoder for fracture anomaly detection.

## 📊 Dataset
The dataset used in this project is the **Fractured Multi-Region X-ray Data** from Kaggle.
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data)
- Dataset structure must be placed in `Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/` with `train`, `val`, `test` splits containing `fractured` and `not fractured` categories.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Eakkachad/bone_recog.git
   cd bone_recog
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For PyTorch, ensure you have the correct version for your CUDA toolkit by following instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/))*

3. **Download SAM 2 Weights:**
   Ensure you have the SAM 2 weights (`sam2.1_b.pt` or similar) downloaded to the root directory for the mask generation step.

## 🚀 Pipeline Workflow

The project is structured numerically to follow a logical execution pipeline:

### 1. Mask Generation (`01_generate_masks.py`)
Uses ultralytics YOLO/SAM module to automatically create binary masks from the X-ray dataset.
```bash
python 01_generate_masks.py
```

### 2. Bone Segmentation Training (`02_train_unet.py`)
Trains a ResNet34-based U-Net model on the generated masks and raw images using `segmentation_models_pytorch`.
```bash
python 02_train_unet.py
```

### 3. Visualizations (`03_`)
Scripts for visualizing SAM generated masks and U-Net results to ensure quality. Includes `03_visualize_sam.py`, `03_visualize_results.py`, and `03_visualization.ipynb`.

### 4. Anomaly Detection Training (`04_train_ae.py`)
Trains an Autoencoder (AE) model on the "not fractured" images to learn the representation of healthy bones. This will be used to detect fractures as anomalies.
```bash
python 04_train_ae.py
```

### 5. Prediction System (`05_predict_system.py`)
The main prediction pipeline that can take a new X-ray image and output whether a fracture is detected using the trained models.

### 6. Evaluation (`06_evaluate_report.py`)
Evaluates the entire system on the test set and generates a comprehensive evaluation report (metrics like Accuracy, F1-Score, and confusion matrix).
```bash
python 06_evaluate_report.py
```

## 📜 License
The code in this repository is open-sourced. The dataset is provided under the Open Data Commons Attribution License (ODC-By) v1.0. 
