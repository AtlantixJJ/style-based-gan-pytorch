import torch
import os
from os.path import join as osj
from PIL import Image
from torchvision import transforms
import numpy as np
import utils


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image only dataset
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
    Image and segmentation pair dataset
    """
    def __init__(self, root, size=(1024, 1024), image_dir="train", label_dir="label", idmap=None):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.root = root
        self.image_dir = osj(root, image_dir)
        self.label_dir = osj(root, label_dir)
        self.idmap = idmap

        self.normal_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imagefiles = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(self.image_dir) if files], [])
        self.imagefiles.sort()
        self.labelfiles = sum([[file for file in files if ".png" in file] for path, dirs, files in os.walk(self.label_dir) if files], [])
        self.labelfiles.sort()

        self.rng = np.random.RandomState(1116)
        self.indice = np.arange(0, len(self.imagefiles))
        self.reset()

    def reset(self):
        self.rng.shuffle(self.indice)

    def transform(self, image, label):
        image_t = self.normal_transform(image)
        label = np.asarray(label.resize(self.size)).copy()
        if self.idmap is not None:
            label = self.idmap(label)
        label_t = torch.from_numpy(label).long().unsqueeze(0)
        if torch.rand(1).numpy()[0] > 0.5:
            image_t = torch.flip(image_t, (2,))
            label_t = torch.flip(label_t, (2,))
        return image_t, label_t

    def __getitem__(self, idx):
        image = utils.pil_read(self.image_dir + "/" + self.imagefiles[idx])
        label = utils.pil_read(self.label_dir + "/" + self.labelfiles[idx])
        image_t, label_t = self.transform(image, label)
        
        return image_t, label_t
    
    def __len__(self):
        return len(self.imagefiles)
    
    def __repr__(self):
        strs = "\n"
        strs += f"=> Root {self.root}\n"
        strs += f"=> Image dir {self.image_dir}\n"
        strs += f"=> Label dir {self.label_dir}\n"
        strs += f"=> Number samples {len(self)}\n"
        return strs


class CollectedDataset(torch.utils.data.Dataset):
    def __init__(self, root, size=(1024, 1024), keys=["origin_latent", "origin_noise", "image_stroke", "image_mask", "label_stroke", "label_mask"]):
        self.root = root
        self.size = size
        self.keys = keys
        self.dic = utils.list_collect_data(root, keys)
        self.normal_transform = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.ToTensor()])

    def __str__(self):
        s = "\n"
        s += f"=> Collected dataset at {self.root}\n"
        s += f"=> Length {len(self)}\n"
        s += f"=> Data keys: {' '.join(self.keys)}\n"
        return s

    def __len__(self):
        return len(self.dic[self.keys[0]])

    def __getitem__(self, idx):
        data_dic = {k: v[idx] for k, v in self.dic.items()}
        for k in self.keys:
            name = data_dic[k]
            if ".npy" in name:
                vec = torch.from_numpy(np.load(data_dic[k]))
                if "latent" in name:
                    vec = vec[0]
                data_dic[k] = vec
            elif ".png" in name:
                img = self.normal_transform(utils.pil_read(data_dic[k]))
                if "label_stroke" in name:
                    img = utils.celeba_rgb2label((img * 255).long())
                if "image_stroke" in name:
                    img = img * 2 - 1
                if "mask" in name:
                    img = img[0:1]
                data_dic[k] = img
        return data_dic

class LatentSegmentationDataset(torch.utils.data.Dataset):
    """
    Reconstructed latent and segmentation dataset.
    """
    def __init__(self, latent_dir=None, noise_dir=None, image_dir=None, seg_dir=None, n_class=19):
        super(LatentSegmentationDataset, self).__init__()
        self.n_class = n_class
        self.latent_dir = latent_dir
        self.noise_dir = noise_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.latent_files = [f for f in os.listdir(self.latent_dir) if ".npy" in f]
        self.latent_files.sort()

        self.rng = np.random.RandomState(1116)
        self.indice = np.arange(0, len(self.latent_files))
        self.reset()

    def reset(self):
        self.rng.shuffle(self.indice)

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, index):
        index = self.indice[index]
        name = self.latent_files[index]
        latent_path = osj(self.latent_dir, name)
        latent = np.load(latent_path)
        seg_path = osj(self.seg_dir, name.replace(".npy", ".png"))
        label = utils.imread(seg_path).copy()
        if self.noise_dir is not None:
            noise_path = osj(self.noise_dir, name)
            noise = np.load(noise_path, allow_pickle=True)
        else:
            noise = 0
        if self.image_dir is not None:
            image_path = osj(self.image_dir, name.replace(".npy", ".jpg"))
            image = utils.imread(image_path).copy()
        else:
            image = 0
        return latent, noise, image, label
    
    def __str__(self):
        strs =  "=> Latent segmentation dataset\n"
        strs += f"=> Number of samples: {len(self)}\n"
        return strs