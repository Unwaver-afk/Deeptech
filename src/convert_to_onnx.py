import torch
from transformers import Dinov2ForImageClassification
import os

MODEL_PATH = "./hackathon_model" # Your saved model path
ONNX_PATH = "WaferSense_Model.onnx"

def convert():
    print("🔄 Loading PyTorch Model...")
    try:
        model = Dinov2ForImageClassification.from_pretrained(MODEL_PATH)
        model.eval()
    except:
        print("❌ Error: Run training first to generate ./hackathon_model")
        return

    # Create dummy input (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"⚙️ Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH, 
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    file_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"✅ ONNX Model Saved! Size: {file_size:.2f} MB")

if __name__ == "__main__":
    convert()