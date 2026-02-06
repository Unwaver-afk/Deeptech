import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# --- CONFIGURATION ---
DATA_PATH = "Wafer_Map_Datasets.npz"
FIGURES_DIR = "figures"

# Ensure output directory exists
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
    print(f"📂 Created '{FIGURES_DIR}' directory.")

# --- LOAD DATA ---
if not os.path.exists(DATA_PATH):
    # Try looking one level up if running from src/
    if os.path.exists(os.path.join("..", DATA_PATH)):
        DATA_PATH = os.path.join("..", DATA_PATH)
    else:
        print(f"❌ Error: {DATA_PATH} not found.")
        exit()

print(f"⏳ Loading {DATA_PATH}...")
data = np.load(DATA_PATH)
raw_imgs = data['arr_0']
raw_lbls = data['arr_1']

# --- FIND EXAMPLE WAFER ---
print("🔍 Searching for a clear defect example...")
idx = 100
while np.sum(raw_lbls[idx]) == 0 and idx < len(raw_lbls) - 1:
    idx += 1
if np.sum(raw_lbls[idx]) == 0: idx = len(raw_lbls) - 1
example_wafer = raw_imgs[idx]
print(f"✅ Using Wafer Index {idx} for visualizations.")


# --- FIGURE 1: HUMAN VS MACHINE VIEW ---
def plot_human_vs_machine():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. Human View (Teal/Navy)
    cmap_human = ListedColormap(['white', '#8cd2d2', '#1e3282']) 
    axes[0].imshow(example_wafer, cmap=cmap_human, interpolation='nearest')
    axes[0].set_title("Operator View (UI)\nOptimized for Human Comfort", fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # 2. Machine View (Red/Gray)
    cmap_machine = ListedColormap(['black', 'gray', 'red']) 
    axes[1].imshow(example_wafer, cmap=cmap_machine, interpolation='nearest')
    axes[1].set_title("Model Input (ViT)\nOptimized for Signal Contrast", fontsize=12, weight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "human_vs_machine.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")


# --- FIGURE 2: PREPROCESSING PIPELINE ---
def plot_pipeline():
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.text(0.5, 0.5, "[[0, 0, 1, 0],\n [0, 2, 1, 0],\n ... ]\n\n(Sparse Matrix)", 
             ha='center', va='center', fontsize=14, fontfamily='monospace')
    ax1.set_title("1. Raw Data (Categorical)", weight='bold')
    ax1.axis('off')
    
    ax_arrow = fig.add_subplot(1, 3, 2)
    ax_arrow.text(0.5, 0.5, "RGB Mapping\nFeature Eng. ➡️", ha='center', va='center', fontsize=12, weight='bold')
    ax_arrow.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3)
    cmap_machine = ListedColormap(['black', 'gray', 'red'])
    ax3.imshow(example_wafer, cmap=cmap_machine, interpolation='nearest')
    ax3.set_title("3. Vision Tensor (224x224)", weight='bold')
    ax3.axis('off')
    
    save_path = os.path.join(FIGURES_DIR, "preprocessing_pipeline.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")


# --- FIGURE 3: RAW DATASET DISTRIBUTION (Total Available) ---
def plot_distribution():
    total = len(raw_lbls)
    defects = np.sum([1 if np.sum(x) > 0 else 0 for x in raw_lbls])
    clean = total - defects
    
    labels = ['Clean Wafers', 'Defected Wafers']
    counts = [clean, defects]
    colors = ['#8cd2d2', '#c62828'] # Teal, Red
    
    plt.figure(figsize=(7, 7))
    if total > 0:
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
                textprops={'fontsize': 12, 'weight': 'bold'}, shadow=True)
    else:
        plt.text(0.5, 0.5, "Dataset Empty", ha='center')

    plt.title(f"Raw Dataset Distribution (Before Balancing)\n(Total: {total:,})", fontsize=14, weight='bold')
    
    save_path = os.path.join(FIGURES_DIR, "dataset_balance.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")


# --- FIGURE 4: EXPERIMENTAL DATA SPLIT (Balanced Subset) ---
# This is the important one for "showing how the model is trained"
def plot_training_split():
    # 1. Calculate Real Counts (Same logic as train.py)
    binary_lbls = np.array([1 if np.sum(l) > 0 else 0 for l in raw_lbls])
    clean_idx = np.where(binary_lbls == 0)[0]
    defect_idx = np.where(binary_lbls == 1)[0]
    
    # 2. Simulate the Balancing (Undersampling Majority)
    # The model uses min(clean, defect) * 2
    min_len = min(len(clean_idx), len(defect_idx))
    balanced_total = min_len * 2
    
    if balanced_total == 0:
        print("⚠️ Cannot plot data split: Dataset seems empty or missing a class.")
        return

    # 3. Calculate 70/15/15 Split on the BALANCED Set
    train_size = int(balanced_total * 0.70)
    val_size = int(balanced_total * 0.15)
    test_size = balanced_total - train_size - val_size
    
    sizes = [train_size, val_size, test_size]
    labels = ['Training (70%)', 'Validation (15%)', 'Testing (15%)']
    colors = ['#2e7d32', '#fdd835', '#1565c0'] # Green, Yellow, Blue
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=(8, 8))
    
    # Custom label function to show Count AND Percentage
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%\n({v:,} imgs)'.format(p=pct, v=val)
        return my_autopct

    plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), 
            startangle=90, colors=colors, explode=explode, 
            textprops={'fontsize': 12, 'weight': 'bold'}, shadow=True)
    
    plt.title(f"Actual Training Data (Balanced Subset)\nTotal Used: {balanced_total:,} images", fontsize=14, weight='bold')
    
    save_path = os.path.join(FIGURES_DIR, "data_split.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path} (Shows balanced counts used for training)")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n🎨 Generating Project Figures...")
    print("="*40)
    
    plot_human_vs_machine()
    plot_pipeline()
    plot_distribution()  # Shows the huge raw dataset
    plot_training_split() # Shows the actual balanced subset used for training
    
    print("="*40)
    print(f"✨ Success! All figures saved to: {os.path.abspath(FIGURES_DIR)}")