import torch
import os
from PIL import Image

class SimpleDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """
    def __init__(self, data_path, size, transform=None):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB").resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.files)