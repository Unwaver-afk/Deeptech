import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Dinov2ForImageClassification, AutoImageProcessor
from PIL import Image
from torchvision import transforms
import random
import os

# --- 1. Hardware Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. Load Real Data ---
print("Initializing Production Line Simulation...")
data_file_path = 'Wafer_Map_Datasets.npz'

if not os.path.exists(data_file_path):
    print(f"❌ Error: Could not find '{data_file_path}'. Please check the file name and location.")
    exit()

data = np.load(data_file_path)
raw_imgs = data['arr_0']
raw_lbls = data['arr_1']

# --- 3. Setup Model ---
model_path = "./hackathon_model"
try:
    model = Dinov2ForImageClassification.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
except Exception as e:
    print(f"❌ Error: Could not find 'hackathon_model' folder or load model. Details: {e}")
    exit()

# Model Transform
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# --- ASSETS FOLDER SETUP ---
ASSETS_BASE = os.path.join("demo", "assets")
CLEAN_FOLDER = os.path.join(ASSETS_BASE, "clean")
DEFECTED_FOLDER = os.path.join(ASSETS_BASE, "defected")

# Create all necessary folders
for folder in [ASSETS_BASE, CLEAN_FOLDER, DEFECTED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_random_wafer():
    idx = random.randint(0, len(raw_imgs) - 1)
    img_array = raw_imgs[idx]
    label_vec = raw_lbls[idx]
    
    # Ground Truth
    actual_status = "Defected" if np.sum(label_vec) > 0 else "Clean"
    
    h, w = img_array.shape
    
    # --- A. MODEL INPUT (Hidden) ---
    rgb_model = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_model[img_array==1] = [128,128,128] 
    rgb_model[img_array==2] = [255,0,0]     
    pil_model = Image.fromarray(rgb_model)
    model_input = transform(pil_model).unsqueeze(0).to(device)

    # --- B. DISPLAY IMAGE (Research Paper Style) ---
    rgb_display = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 0 = Background -> White
    rgb_display[img_array==0] = [255, 255, 255]
    
    # 1 = Normal Die -> Soft Teal
    rgb_display[img_array==1] = [140, 210, 210] 
    
    # 2 = Defect -> Dark Navy Blue
    rgb_display[img_array==2] = [30, 50, 130] 
    
    pil_display = Image.fromarray(rgb_display)
    pil_display = pil_display.resize((400, 400), resample=Image.NEAREST)
    
    return pil_display, model_input, actual_status, idx

def run_simulation(counter, total):
    display_img, model_input, actual, idx = get_random_wafer()
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=model_input)
        logits = outputs.logits
        prediction_idx = torch.argmax(logits, dim=1).item()
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence_score = probs[0][prediction_idx].item()
        
    if prediction_idx == 0:
        status_box = "PASSED"
        text_color = '#2e7d32' 
        box_color = '#e8f5e9'  
        edge_color = '#2e7d32'
        target_folder = CLEAN_FOLDER # Save to clean
    else:
        status_box = "REJECTED"
        text_color = '#c62828' 
        box_color = '#ffebee'  
        edge_color = '#c62828'
        target_folder = DEFECTED_FOLDER # Save to defected

    # --- DASHBOARD ---
    plt.figure(figsize=(10, 5), facecolor='white')
    
    # 1. Wafer Map
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(display_img)
    ax1.set_title(f"Wafer Index: {idx}", fontsize=14, color='black', weight='bold')
    ax1.axis('off')
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # 2. Results Panel
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    ax2.text(0.5, 0.85, "AUTOMATED INSPECTION", ha='center', va='center', fontsize=12, color='gray', weight='bold')
    
    # Status Box
    ax2.text(0.5, 0.6, status_box, ha='center', va='center', fontsize=32, weight='bold', color=text_color,
             bbox=dict(facecolor=box_color, boxstyle='round,pad=0.5', edgecolor=edge_color, linewidth=2))
    
    ax2.text(0.5, 0.35, f"Confidence: {confidence_score:.2%}", ha='center', va='center', fontsize=12, color='black')
    ax2.text(0.5, 0.25, f"Ground Truth: {actual}", ha='center', va='center', fontsize=11, color='gray')
    ax2.text(0.5, 0.1, "■ Normal (Teal)   ■ Defect (Navy)", ha='center', va='center', fontsize=10, color='black')

    plt.tight_layout()
    
    # --- AUTO SAVE TO SUBFOLDER ---
    save_path = os.path.join(target_folder, f"{idx}.jpeg")
    plt.savefig(save_path)
    
    print(f"[{counter}/{total}] Saved: {save_path} | Result: {status_box}")
    
    plt.draw()
    plt.pause(0.5) 
    plt.close()

if __name__ == "__main__":
    try:
        user_input = input("Enter the number of demo images to generate: ")
        num_images = int(user_input)
    except ValueError:
        print("Invalid input. Defaulting to 1 image.")
        num_images = 1

    print(f"🚀 Starting generation of {num_images} assets...")
    print(f"📂 Clean images will go to: {CLEAN_FOLDER}")
    print(f"📂 Defected images will go to: {DEFECTED_FOLDER}")
    
    for i in range(num_images):
        run_simulation(i + 1, num_images)
        
    print(f"\n✅ Batch generation complete!")