import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time  # <--- NEW
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef  # <--- NEW
)
from transformers import Dinov2ForImageClassification, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
MODEL_PATH = "./hackathon_model"
DATA_PATH = "Wafer_Map_Datasets.npz" 
RESULTS_DIR = "results"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
TEST_SAMPLES = 2000 

# --- SETUP ---
print(f"🚀 Initializing Diamond-Standard Evaluation on {DEVICE}...")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found.")
    exit()

data = np.load(DATA_PATH)
raw_imgs = data['arr_0']
raw_lbls = data['arr_1']

try:
    model = Dinov2ForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
except Exception as e:
    print(f"❌ Model Error: {e}")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

class EvalDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_array = self.images[idx]
        label = 1 if np.sum(self.labels[idx]) > 0 else 0
        h, w = img_array.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[img_array==1] = [128,128,128]
        rgb[img_array==2] = [255,0,0]
        img = Image.fromarray(rgb)
        return self.transform(img), label

indices = np.random.choice(len(raw_imgs), TEST_SAMPLES, replace=False)
test_imgs = raw_imgs[indices]
test_lbls = raw_lbls[indices]
dataset = EvalDataset(test_imgs, test_lbls, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# --- PHASE 1: INFERENCE & LATENCY CHECK ---
print("📊 Running Inference & Benchmarking Speed...")
all_preds = []
all_labels = []
all_probs = []

start_time = time.time()  # <--- START TIMER

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(pixel_values=imgs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        defect_prob = probs[:, 1].cpu().numpy()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        all_probs.extend(defect_prob)
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

end_time = time.time()  # <--- STOP TIMER

# Calculate Latency
total_time = end_time - start_time
latency_ms = (total_time / TEST_SAMPLES) * 1000
fps = TEST_SAMPLES / total_time
print(f"⚡ Speed: {latency_ms:.2f} ms/wafer ({fps:.0f} FPS)")

# --- PHASE 2: GENERATE METRICS ---

# Standard Metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)  # <--- NEW METRIC

print(f"\n✅ Accuracy:  {acc:.2%}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"💎 MCC:       {mcc:.4f} (The 'Diamond' Metric)")

# Save JSON
metrics_dict = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "mcc": mcc,
    "latency_ms": latency_ms,
    "fps": fps,
    "samples": TEST_SAMPLES
}
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)
print("📄 Saved metrics.json")

# ... (Rest of plotting code: Confusion Matrix, ROC, etc. remains same) ...
# Just copy the plotting parts from the previous script here!
# For brevity, I assume you have the plotting code from the previous response.
# -------------------------------------------------------------------------
# [PASTE THE PLOTTING CODE FOR CONFUSION MATRIX, ROC, ETC. HERE]
# -------------------------------------------------------------------------

# (Quick re-paste of the plots just in case you need them)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Defect'], yticklabels=['Clean', 'Defect'])
plt.title(f'Confusion Matrix (MCC={mcc:.3f})') # Added MCC to title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(all_probs, bins=20, kde=True, color='purple')
plt.title('Model Confidence Distribution')
plt.axvline(0.5, color='red', linestyle='--')
plt.savefig(os.path.join(RESULTS_DIR, "confidence_histogram.png"))
plt.close()

print(f"\n✅ ALL RESULTS SAVED TO: {os.path.abspath(RESULTS_DIR)}")