import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.dataset import WaferDataset
from src.model import load_model
from src.preprocess import get_transforms

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Data (Test Set Logic - Simplified)
    # In real use, ensure this matches your specific Test Split
    data = np.load('data_mutil_label.npz')
    # ... [Insert your splitting logic here to get X_test, y_test] ...
    # For demo, let's pretend we have X_test and y_test ready
    # YOU MUST POPULATE THIS PART WITH YOUR TEST DATA SPLIT LOGIC
    
    # Dummy placeholder for structure (REPLACE THIS)
    X_test = data['arr_0'][:100] 
    y_test = data['arr_1'][:100]

    transform = get_transforms()
    ds = WaferDataset(X_test, y_test, transform=transform)
    loader = DataLoader(ds, batch_size=32)
    
    model = load_model("./wafer_model_v1", device=device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(pixel_values=imgs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- METRICS GENERATION ---
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Defect'], yticklabels=['Clean', 'Defect'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    print("✅ Confusion Matrix saved to results/confusion_matrix.png")

    # 2. Detailed Metrics (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=['Clean', 'Defect'], output_dict=True)
    
    # Save to JSON for README
    with open('results/metrics.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("✅ Extended Metrics saved to results/metrics.json")
    
    # Print readable report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Clean', 'Defect']))

if __name__ == "__main__":
    evaluate()