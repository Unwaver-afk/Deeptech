import torch
import numpy as np
import onnxruntime as ort
from transformers import Dinov2ForImageClassification

# ---------- CONFIG ----------
PYTORCH_MODEL_DIR = "hackathon_model"
ONNX_MODEL_PATH = "onnx/WaferSense_Model.onnx"
DEVICE = "cpu"  # use cpu for deterministic comparison
# ----------------------------

# Load PyTorch model
torch_model = Dinov2ForImageClassification.from_pretrained(
    PYTORCH_MODEL_DIR
)
torch_model.eval()
torch_model.to(DEVICE)

# Create a fixed random input (seeded)
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

# PyTorch inference
with torch.no_grad():
    torch_output = torch_model(dummy_input).logits
torch_output_np = torch_output.cpu().numpy()

# Load ONNX model
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ONNX inference
onnx_output = session.run(
    None,
    {input_name: dummy_input.cpu().numpy()}
)[0]

# ---------- COMPARISON ----------
abs_diff = np.abs(torch_output_np - onnx_output)
max_diff = abs_diff.max()
mean_diff = abs_diff.mean()

print("Max absolute difference:", max_diff)
print("Mean absolute difference:", mean_diff)

# Sanity check
if max_diff < 1e-3:
    print("✅ ONNX and PyTorch outputs MATCH")
else:
    print("❌ Mismatch detected")
