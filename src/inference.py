import torch
import numpy as np
from PIL import Image
import argparse
import sys
import os

# Add project root to path so we can run this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_model
from src.preprocess import get_transforms, to_rgb_model_input

# Config
MODEL_PATH = "./hackathon_model"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def predict_wafer(image_input, model, device):
    """
    Runs inference on a single image.
    Args:
        image_input: Can be a PIL Image (RGB) OR a raw Numpy array (52x52).
    Returns:
        prediction_label (str), confidence (float)
    """
    transform = get_transforms()

    # 1. Handle Input Type
    if isinstance(image_input, np.ndarray):
        # If raw wafer map (0/1/2), convert to Model RGB (Red/Gray)
        pil_image = to_rgb_model_input(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input
    else:
        raise ValueError("Unsupported image type. Use Numpy Array or PIL Image.")

    # 2. Preprocess
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 3. Inference
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    label = "Defect" if pred_idx.item() == 1 else "Clean"
    return label, confidence.item()

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Run AI Inference on a Single Wafer")
    parser.add_argument("--random", action="store_true", help="Pick a random wafer from dataset to test")
    # In a real app, you might add: parser.add_argument("--image_path", type=str, help="Path to a JPEG file")
    args = parser.parse_args()

    # Load Model
    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH, device=DEVICE)
    except:
        print("❌ Model not found. Please run train.py first.")
        return

    # Get Data (For Simulation/Testing)
    if args.random:
        print("🎲 Picking random sample from dataset...")
        try:
            data = np.load('Wafer_Map_Datasets.npz')
            raw_imgs = data['arr_0']
            raw_lbls = data['arr_1']
            
            idx = np.random.randint(0, len(raw_imgs))
            wafer_img = raw_imgs[idx]
            ground_truth = "Defect" if np.sum(raw_lbls[idx]) > 0 else "Clean"
            
            # RUN PREDICTION
            pred_label, conf = predict_wafer(wafer_img, model, DEVICE)
            
            print("\n" + "="*30)
            print(f"🔍 INFERENCE RESULTS (Index {idx})")
            print("="*30)
            print(f"✅ Prediction:   {pred_label}")
            print(f"📊 Confidence:   {conf:.4%}")
            print(f"📝 Ground Truth: {ground_truth}")
            
            if pred_label == ground_truth:
                print("\nResult: CORRECT match.")
            else:
                print("\nResult: MISMATCH.")
                
        except FileNotFoundError:
            print("❌ Error: data_mutil_label.npz not found.")
    else:
        print("ℹ️  Usage: python src/inference.py --random")
        print("   (This runs a test on a random sample from your database)")

if __name__ == "__main__":
    main()