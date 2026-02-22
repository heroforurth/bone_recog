
"""
06_evaluate_report.py
Evaluates the Bone vs Not-Bone detection system.
- Bone images: from Bone_Fracture_Binary_Classification (test split)
- Not-bone images: from not_bone/ folder
Outputs: evaluation_report.txt + evaluation_report.png
"""

import os
import cv2
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from anomaly_model import ConvAutoencoder

# ── Config ─────────────────────────────────────────────────────────────────
DATA_ROOT     = "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
NOT_BONE_DIR  = "not_bone"
AE_MODEL_PATH = "ae_bone.pth"
THRESHOLD_FILE= "ae_threshold.txt"
AE_IMG_SIZE   = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BONE_IMGS = 200   # cap so it runs fast; set None for all
REPORT_TXT    = "evaluation_report.txt"
REPORT_PNG    = "evaluation_report.png"
# ───────────────────────────────────────────────────────────────────────────

def load_image(path, size):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    t   = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return t, img

def collect_bone_paths():
    paths = []
    for split in ["train", "val", "test"]:
        for cat in ["fractured", "not fractured"]:
            d = os.path.join(DATA_ROOT, split, cat)
            if not os.path.exists(d):
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                paths.extend(glob.glob(os.path.join(d, ext)))
    return paths

def run_evaluation():
    print(f"Device: {DEVICE}")

    # ── Load model ──────────────────────────────────────────────────────────
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae.eval()

    with open(THRESHOLD_FILE) as f:
        threshold = float(f.read().strip())
    print(f"Threshold: {threshold:.6f}")

    # ── Collect paths ───────────────────────────────────────────────────────
    bone_paths = collect_bone_paths()
    if MAX_BONE_IMGS and len(bone_paths) > MAX_BONE_IMGS:
        np.random.seed(42)
        bone_paths = list(np.random.choice(bone_paths, MAX_BONE_IMGS, replace=False))

    not_bone_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        not_bone_paths.extend(glob.glob(os.path.join(NOT_BONE_DIR, ext)))

    print(f"Bone images  : {len(bone_paths)}")
    print(f"Not-bone imgs: {len(not_bone_paths)}")

    # ── Inference ───────────────────────────────────────────────────────────
    results = []   # (filename, true_label, pred_label, mse)
    sample_imgs = {"bone_correct": [], "bone_wrong": [],
                   "notbone_correct": [], "notbone_wrong": []}

    def infer(path, true_label):
        out = load_image(path, AE_IMG_SIZE)
        if out is None:
            return
        t, rgb = out
        t = t.to(DEVICE)
        with torch.no_grad():
            recon = ae(t)
            mse   = torch.mean((t - recon) ** 2).item()
        pred = "bone" if mse <= threshold else "not_bone"
        results.append((os.path.basename(path), true_label, pred, mse))

        # collect samples for visual grid
        key = f"{'bone' if true_label=='bone' else 'notbone'}_{'correct' if pred==true_label else 'wrong'}"
        if len(sample_imgs[key]) < 3:
            sample_imgs[key].append((rgb, mse, pred))

    for p in bone_paths:
        infer(p, "bone")
    for p in not_bone_paths:
        infer(p, "not_bone")

    # ── Metrics ─────────────────────────────────────────────────────────────
    y_true = [1 if r[1]=="bone" else 0 for r in results]
    y_pred = [1 if r[2]=="bone" else 0 for r in results]

    TP = sum(t==1 and p==1 for t,p in zip(y_true,y_pred))
    TN = sum(t==0 and p==0 for t,p in zip(y_true,y_pred))
    FP = sum(t==0 and p==1 for t,p in zip(y_true,y_pred))
    FN = sum(t==1 and p==0 for t,p in zip(y_true,y_pred))

    total     = len(results)
    accuracy  = (TP+TN)/total if total else 0
    precision = TP/(TP+FP) if (TP+FP) else 0
    recall    = TP/(TP+FN) if (TP+FN) else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    # ── Per-image table ─────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  BONE vs NOT-BONE DETECTION — EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"  Anomaly Threshold : {threshold:.6f}")
    lines.append(f"  Total images      : {total}")
    lines.append(f"  Bone images       : {sum(1 for r in results if r[1]=='bone')}")
    lines.append(f"  Not-bone images   : {sum(1 for r in results if r[1]=='not_bone')}")
    lines.append("")
    lines.append("  METRICS")
    lines.append(f"  Accuracy          : {accuracy*100:.2f}%")
    lines.append(f"  Precision (bone)  : {precision*100:.2f}%")
    lines.append(f"  Recall    (bone)  : {recall*100:.2f}%")
    lines.append(f"  F1 Score          : {f1*100:.2f}%")
    lines.append("")
    lines.append("  CONFUSION MATRIX")
    lines.append(f"  {'':20s}  Pred: Bone  Pred: Not-Bone")
    lines.append(f"  {'True: Bone':20s}  {TP:10d}  {FN:14d}")
    lines.append(f"  {'True: Not-Bone':20s}  {FP:10d}  {TN:14d}")
    lines.append("")
    lines.append("  PER-IMAGE RESULTS")
    lines.append(f"  {'Filename':<35} {'True':10} {'Predicted':12} {'MSE':>10}  {'OK?':5}")
    lines.append("  " + "-"*68)
    for fname, true_lbl, pred_lbl, mse in sorted(results, key=lambda x: x[1]):
        ok = "✓" if true_lbl == pred_lbl else "✗"
        lines.append(f"  {fname:<35} {true_lbl:<10} {pred_lbl:<12} {mse:>10.6f}  {ok}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_TXT, "w") as f:
        f.write(report_text)
    print(f"\nReport saved → {REPORT_TXT}")

    # ── Visual Report ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # -- Confusion matrix heatmap --
    ax_cm = fig.add_subplot(gs[0, :2])
    cm = np.array([[TP, FN], [FP, TN]])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Pred: Bone","Pred: Not-Bone"], color="white")
    ax_cm.set_yticklabels(["True: Bone","True: Not-Bone"], color="white")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                       fontsize=18, color="black" if cm[i,j]>cm.max()/2 else "white")
    ax_cm.set_title("Confusion Matrix", color="white", fontsize=13)
    ax_cm.tick_params(colors="white")

    # -- Metrics bar chart --
    ax_bar = fig.add_subplot(gs[0, 2:])
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
    colors  = ["#4cc9f0","#4361ee","#3a0ca3","#7209b7"]
    bars = ax_bar.bar(metrics.keys(), [v*100 for v in metrics.values()], color=colors)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Score (%)", color="white")
    ax_bar.set_title("Performance Metrics", color="white", fontsize=13)
    ax_bar.tick_params(colors="white")
    ax_bar.set_facecolor("#16213e")
    for bar, val in zip(bars, metrics.values()):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{val*100:.1f}%", ha="center", color="white", fontsize=11)

    # -- MSE distribution --
    ax_hist = fig.add_subplot(gs[1, :])
    bone_mses    = [r[3] for r in results if r[1]=="bone"]
    notbone_mses = [r[3] for r in results if r[1]=="not_bone"]
    if bone_mses:
        ax_hist.hist(bone_mses, bins=30, alpha=0.7, color="#4cc9f0", label="Bone")
    if notbone_mses:
        ax_hist.hist(notbone_mses, bins=10, alpha=0.7, color="#f72585", label="Not-Bone")
    ax_hist.axvline(threshold, color="yellow", linestyle="--", linewidth=2,
                    label=f"Threshold ({threshold:.5f})")
    ax_hist.set_xlabel("Reconstruction MSE", color="white")
    ax_hist.set_ylabel("Count", color="white")
    ax_hist.set_title("MSE Distribution (Bone vs Not-Bone)", color="white", fontsize=13)
    ax_hist.tick_params(colors="white")
    ax_hist.set_facecolor("#16213e")
    legend = ax_hist.legend(facecolor="#16213e", labelcolor="white")

    # -- Sample images (bottom row) --
    sample_keys = [("bone_correct","Bone ✓","#4cc9f0"),
                   ("bone_wrong","Bone ✗","#f72585"),
                   ("notbone_correct","Not-Bone ✓","#4cc9f0"),
                   ("notbone_wrong","Not-Bone ✗","#f72585")]
    for col_idx, (key, label, color) in enumerate(sample_keys):
        imgs = sample_imgs[key]
        if imgs:
            rgb, mse, pred = imgs[0]
            ax = fig.add_subplot(gs[2, col_idx])
            ax.imshow(rgb)
            ax.set_title(f"{label}\nMSE:{mse:.5f}", color=color, fontsize=9)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3)

    fig.suptitle("Bone vs Not-Bone Detection — Evaluation Report",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    plt.savefig(REPORT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Visual report saved → {REPORT_PNG}")

if __name__ == "__main__":
    run_evaluation()
