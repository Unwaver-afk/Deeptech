import torch
import numpy as np
from torch.utils.data import Dataset
from src.preprocess import to_rgb_model_input

class WaferDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = self.images[idx]
        
        # Logic: If label vector has any 1s, it's a defect
        is_defect = 1 if np.sum(self.labels[idx]) > 0 else 0
        
        # Use our helper function
        img = to_rgb_model_input(img_array)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(is_defect, dtype=torch.long)