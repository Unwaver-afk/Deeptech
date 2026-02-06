import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import Dinov2ForImageClassification
from torch.optim import AdamW

# --- IMPORTS FROM YOUR SRC FOLDER ---
# This keeps your project modular as shown in your screenshots
try:
    from src.dataset import WaferDataset
    from src.preprocess import get_transforms
except ImportError:
    # Fallback if running script directly inside src/
    from dataset import WaferDataset
    from preprocess import get_transforms

# --- CONFIGURATION ---
DATA_FILE = 'Wafer_Map_Datasets.npz'
MODEL_SAVE_PATH = "./hackathon_model"
MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

def main():
    # --- 1. M4 Hardware Check ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple M4 Neural Engine (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")

    # --- 2. Data Loading ---
    # Robust path handling to find the file from root or src
    if os.path.exists(DATA_FILE):
        data_path = DATA_FILE
    elif os.path.exists(os.path.join("..", DATA_FILE)):
        data_path = os.path.join("..", DATA_FILE)
    else:
        print(f"❌ Error: Could not find {DATA_FILE}")
        return

    print(f"📂 Loading data from {data_path}...")
    data = np.load(data_path)
    raw_images = data['arr_0']
    raw_labels = data['arr_1']

    # --- 3. Balancing Logic (Robust 50/50) ---
    # 0 = Clean, 1 = Defect
    binary_labels = np.array([1 if np.sum(l) > 0 else 0 for l in raw_labels])

    clean_indices = np.where(binary_labels == 0)[0]
    defect_indices = np.where(binary_labels == 1)[0]

    print(f"   Total Clean Available:    {len(clean_indices)}")
    print(f"   Total Defect Available:   {len(defect_indices)}")

    # SAFETY FIX: Limit samples to the smaller class count to prevent crashing
    # If you have 25k clean and 1k defects, we pick 1k of each.
    min_samples = min(len(clean_indices), len(defect_indices))
    
    selected_clean = np.random.choice(clean_indices, min_samples, replace=False)
    selected_defect = np.random.choice(defect_indices, min_samples, replace=False)

    final_indices = np.concatenate([selected_clean, selected_defect])
    
    # Shuffle the combined indices
    np.random.shuffle(final_indices)

    final_images = raw_images[final_indices]
    final_labels = raw_labels[final_indices] # We keep original labels for dataset compatibility

    # Recalculate binary labels for splitting
    final_binary_labels = binary_labels[final_indices]

    print(f"✅ Balanced Dataset Created: {len(final_indices)} images (50% Clean / 50% Defect)")

    # --- 4. Splitting (70% Train / 15% Val / 15% Test) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        final_images, final_labels, test_size=0.3, stratify=final_binary_labels, random_state=42
    )
    
    # Split the 30% temp into 15% Val and 15% Test
    # We need binary labels of X_temp for stratification
    y_temp_binary = np.array([1 if np.sum(l) > 0 else 0 for l in y_temp])
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp_binary, random_state=42
    )

    print(f"   Stats -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- 5. Transforms & Loaders ---
    # Using your modular src/preprocess.py
    transform = get_transforms()

    train_ds = WaferDataset(X_train, y_train, transform=transform)
    val_ds = WaferDataset(X_val, y_val, transform=transform)
    # Note: Test set is usually handled by evaluate.py, but we define it here if needed

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 6. Model Setup ---
    print(f"🏗️  Loading {MODEL_NAME}...")
    model = Dinov2ForImageClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 7. Training Loop ---
    print(f"\n🚀 Starting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Training Step
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Dinov2 expects 'labels' argument for loss calculation
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Optional: Print progress every 10 steps
            if step % 10 == 0 and step > 0:
                print(f"   Epoch {epoch+1} Step {step}: Loss {loss.item():.4f}", end='\r')
        
        # Validation Step
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for v_images, v_labels in val_loader:
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                v_outputs = model(pixel_values=v_images)
                preds = torch.argmax(v_outputs.logits, dim=1)
                val_correct += (preds == v_labels).sum().item()
        
        val_acc = val_correct / len(val_ds)
        avg_loss = total_loss / len(train_loader)
        
        print(f"\n✅ Epoch {epoch+1}/{EPOCHS}: Avg Loss = {avg_loss:.4f} | Val Accuracy = {val_acc:.2%}")

    # --- 8. Save Model ---
    print(f"💾 Saving model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    print("🎉 Done! Ready for demo.")

if __name__ == "__main__":
    main()