import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import utils

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

class ImageSegmentationDataset(torch.utils.data.Dataset):
    """
    Currently label is not available
    """
    def __init__(self, data_path, size):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.root_dir = data_path
        self.image_dir = data_path + "/train"
        self.label_dir = data_path + "/label"

        self.normal_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imagefiles = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(self.image_dir) if files], [])
        self.imagefiles.sort()
        self.labelfiles = sum([[file for file in files if ".png" in file] for path, dirs, files in os.walk(self.label_dir) if files], [])
        self.labelfiles.sort()

    def transform(self, image, label):
        image_t = self.normal_transform(image)
        label = np.asarray(label.resize(self.size))[:, :, 0]
        label_t = torch.from_numpy(label).long()
        print(label_t.min(), label_t.max())
        return image_t, label_t

    def __getitem__(self, idx):
        image = utils.pil_read(self.image_dir + "/" + self.imagefiles[idx])
        label = utils.pil_read(self.label_dir + "/" + self.labelfiles[idx])
        return self.transform(image, label)
    
    def __len__(self):
        return len(self.imagefiles)


class LatentSegmentationDataset(ImageSegmentationDataset):
    def __init__(self, data_path, size):
        super(LatentSegmentationDataset, self).__init__(data_path, size)

        with open(self.root_dir + "/latents.npz", 'rb') as fp:
            latents_np = np.load(fp)
        self.latents = torch.from_numpy(latents_np)
        del latents_np
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        image = utils.pil_read(self.image_dir + "/" + self.imagefiles[idx])
        label = utils.pil_read(self.label_dir + "/" + self.labelfiles[idx])
        image_t, label_t = self.transform(image, label)
        return latent, image_t, label_t