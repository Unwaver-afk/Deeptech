import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor

# Define standard transformation pipeline
def get_transforms(model_name="facebook/dinov2-base"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

def to_rgb_model_input(img_array):
    """
    Converts 0/1/2 sparse matrix to Red/Gray/Black for the AI.
    """
    h, w = img_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[img_array == 1] = [128, 128, 128]  # Gray (Normal)
    rgb[img_array == 2] = [255, 0, 0]      # Red (Defect)
    return Image.fromarray(rgb)

def to_rgb_display(img_array):
    """
    Converts 0/1/2 to White/Teal/Navy for Human UI (Research Paper Style).
    """
    h, w = img_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[img_array == 0] = [255, 255, 255]  # White BG
    rgb[img_array == 1] = [140, 210, 210]  # Teal Normal
    rgb[img_array == 2] = [30, 50, 130]    # Navy Defect
    return Image.fromarray(rgb)