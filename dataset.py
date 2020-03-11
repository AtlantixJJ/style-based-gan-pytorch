import torch
import os, zipfile
from io import BytesIO
from os.path import join as osj
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import utils
import math
from scipy.ndimage.morphology import binary_erosion


class IterableDataset(torch.utils.data.Dataset):
    def __init__(self, iterable):
        super(IterableDataset, self).__init__()
        self.iterable = iterable
    
    def __len__(self):
        return len(self.iterable)

    def __getitem__(self, idx):
        return self.iterable[idx]
        

class CelebAZipDataset(torch.utils.data.Dataset):
    """
    celeba-dataset.zip
    partition: 0 for training, 1 for validation, 2 for testing
    """
    def __init__(self, root, size, partition=0, transform=None):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.root = root
        self.partition = partition
        self.transform = transform
        if self.transform is None and self.partition == 0:
            self.transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ColorJitter(0.05, 0.1, 0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.zipfile = zipfile.ZipFile(f"{root}/celeba-dataset.zip")

        self.files = [f for f in self.zipfile.namelist() if ".jpg" in f]
        self.files.sort()

        self.attr_names, self.attrs = self.read_csv_from_zip(
            "list_attr_celeba.csv")
        _, self.bboxes = self.read_csv_from_zip(
            "list_bbox_celeba.csv")
        _, self.partitions = self.read_csv_from_zip(
            "list_eval_partition.csv")
        _, self.landmarks = self.read_csv_from_zip(
            "list_landmarks_align_celeba.csv")
        self.bboxes = [[int(i) for i in b] for b in self.bboxes]
    
    def __len__(self):
        return len(self.partitions)

    def read_csv_from_zip(self, f):
        lines = self.zipfile.read(f).decode("ascii").split("\r\n")
        head = lines[0]
        content = [l.split(",")[1:] for l in lines[1:]]
        return head[1:], content

    def read_image_from_zip(self, f):
        return Image.open(BytesIO(self.zipfile.read(f)))

    def reset(self):
        pass

    def __getitem__(self, idx):
        image = self.read_image_from_zip(self.files[idx])
        #p = self.bboxes[idx]
        image = image.crop((15, 40, 15 + 148, 40 + 148))
        if self.transform:
            image = self.transform(image)
        return image


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image only dataset
    """
    def __init__(self, root, size, transform=None):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.root = root
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(root) if files], [])
        self.files.sort()

    def reset(self):
        pass

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.root, fpath), "rb") as f:
            img = Image.open(f).convert("RGB").resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.files)
    
    def __str__(self):
        strs = []
        strs.append(f"=> Root: {self.root}")
        strs.append(f"=> Number of samples: {len(self)}")
        strs.append(str(self.transform))
        return "\n".join(strs)


class ImageSegmentationDataset(torch.utils.data.Dataset):
    """
    Image and segmentation pair dataset
    """
    def __init__(self, root, size=(1024, 1024), image_dir="train", label_dir="label", idmap=None, random_flip=True, file_list=None):
        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.root = root
        self.flip = random_flip
        self.image_dir = osj(root, image_dir)
        self.label_dir = osj(root, label_dir)
        self.idmap = idmap
        self.file_list = file_list

        self.normal_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if self.file_list is None:
            self.imagefiles = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(self.image_dir) if files], [])
            self.labelfiles = sum([[file for file in files if ".png" in file] for path, dirs, files in os.walk(self.label_dir) if files], [])
        else:
            files = open(file_list, "r").read().split("\n")
            self.imagefiles = [f"{f}.jpg" for f in files if len(f) > 0]
            self.labelfiles = [f"{f}.png" for f in files if len(f) > 0]

        self.imagefiles.sort()
        self.labelfiles.sort()

        self.rng = np.random.RandomState(1116)
        self.indice = np.arange(0, len(self.imagefiles))
        self.reset()

    def reset(self):
        self.rng.shuffle(self.indice)

    def transform(self, image, label):
        image_t = self.normal_transform(image)
        label = np.asarray(label.resize(self.size, Image.NEAREST)).copy()
        if self.idmap is not None:
            label = self.idmap.mapid(label)
        label_t = torch.from_numpy(label).long().unsqueeze(0)
        if self.flip and torch.rand(1).numpy()[0] > 0.5:
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


class ImageSegmentationPartDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        if image.shape[0] != 1:
            image = image.unsqueeze(0)
        image = utils.tensor2image(image)
        label = utils.torch2numpy(label)[0].astype("uint8")
        parts = utils.image2part_catetory(image, label)
        # preprocessing: numpy uint8 -> torch
        for i in range(len(parts)):
            parts[i][1] = self.transform(Image.fromarray(parts[i][1]))
        return parts


class SegGANDataset(torch.utils.data.Dataset):
    def __init__(self, model, dim=512, tot_num=30000, batch_size=1, save_path=None, device='cuda'):
        self.model = model
        self.dim = dim
        self.tot_num = tot_num
        self.batch_size = batch_size
        self.save_path = save_path
        self.device = device
        self.num_iter = math.ceil(float(self.tot_num) / self.batch_size)

        self.z = torch.Tensor(self.batch_size, self.dim).to(device)
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.system("mkdir %s" % self.save_path)
            os.system("mkdir %s/image" % self.save_path)
            os.system("mkdir %s/label" % self.save_path)

    def __len__(self):
        return self.num_iter

    def save(self, gidx, image, label):
        for i in range(image.shape[0]):
            vutils.save_image((1 + image[i:i+1]) / 2,
                f"{self.save_path}/image/{gidx+i}.jpg")
            utils.imwrite(
                f"{self.save_path}/label/{gidx+i}.png", utils.torch2numpy(label[i]))

    def __getitem__(self, idx):
        z = self.z
        if idx == self.num_iter - 1:
            bs = self.tot_num - self.batch_size * idx
            if bs < self.batch_size:
                z = torch.Tensor(bs, self.dim, device=self.device)
        z.normal_()

        image, seg = self.model(z)
        label = seg.argmax(1)
        image = image.clamp(-1, 1)

        if self.save_path is not None:
            self.save(idx * self.batch_size, image, label)

        return image, label


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
        for key in self.dic.keys():
            s += f"=> {key}: {len(self.dic[key])}\n"
        return s

    def __len__(self):
        return len(self.dic[self.keys[0]])

    def __getitem__(self, idx):
        data_dic = {k: v[idx] for k, v in self.dic.items()}
        for k in self.keys:
            name = data_dic[k]
            if ".npy" in name:
                data = torch.from_numpy(np.load(data_dic[k]))
                if "latent" in name:
                    data = data[0]
            elif ".png" in name:
                img = self.normal_transform(utils.pil_read(data_dic[k]))
                if "label_stroke" in name:
                    data = utils.celeba_rgb2label((img * 255).long())
                elif "image_stroke" in name:
                    data = img * 2 - 1
                elif "image_mask" in name:
                    x = img[0].detach().numpy().astype("uint8")
                    data = binary_erosion(x)
                    data = torch.from_numpy(data).float()
                elif "label_mask" in name:
                    x = img[0].detach().numpy().astype("uint8")
                    data = binary_erosion(x)
                    data = torch.from_numpy(data).float()
            data_dic[k] = data
        return data_dic


class LatentSegmentationDataset(torch.utils.data.Dataset):
    """
    Reconstructed latent and segmentation dataset.
    """
    def __init__(self,
        latent_dir=None, noise_dir=None, image_dir=None, seg_dir=None,
        n_class=19, idmap=None):
        super(LatentSegmentationDataset, self).__init__()
        self.n_class = n_class
        self.latent_dir = latent_dir
        self.noise_dir = noise_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.idmap = idmap
        self.latent_files = [f for f in os.listdir(self.latent_dir) if ".npy" in f]
        self.latent_files.sort()

        #self.rng = np.random.RandomState(1116)
        self.indice = np.arange(0, len(self.latent_files))
        #self.reset()

    def reset(self):
        pass
        #self.rng.shuffle(self.indice)

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, index):
        index = self.indice[index]
        name = self.latent_files[index]
        latent_path = osj(self.latent_dir, name)
        latent = np.load(latent_path)
        seg_path = osj(self.seg_dir, name.replace(".npy", ".png"))
        label = utils.imread(seg_path).copy()
        if self.idmap:
            label = self.idmap(label)
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


if __name__ == "__main__":
    ds = CelebAZipDataset("../datasets/CelebA", 128)
    for idx, img in enumerate(ds):
        print(img.shape)
        vutils.save_image(img, f"{idx}.png", normalize=True)
        if idx > 10:
            break