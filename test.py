import os
from os.path import join as osj
import torch
import argparse
import glob
import numpy as np
import utils
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet
from lib.face_parsing.utils import tensor2label

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir, image_dir, seg_dir):
        super(SegmentationDataset, self).__init__()
        self.latent_dir = latent_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.latent_files = os.listdir(self.latent_dir)
        self.latent_files.sort()

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, index):
        name = self.latent_files[index]
        latent_path = osj(self.latent_dir, name)
        image_path = osj(self.image_dir, name.replace(".npy", ".jpg"))
        seg_path = osj(self.seg_dir, name.replace(".npy", ".png"))
        latent = np.load(latent_path)
        image = utils.imread(image_path)
        segmentation = utils.imread(seg_path)
        return latent, image, segmentation

rootdir = "/home/xujianjin/data/datasets/CelebAMask-HQ/"
ds = SegmentationDataset(
    latent_dir=rootdir+"dlatent5000",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

latent, image, segmentation = ds[10]
latent = torch.from_numpy(latent)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--step", type=int, default=8)
args = parser.parse_args()

cfg = config.config_from_name(args.model)
print(cfg)
generator = StyledGenerator(**cfg)
generator = generator.cuda()

model_files = glob.glob(args.model + "/*.model")
model_files.sort()
print("=> Load from %s" % model_files[-1])
generator.load_state_dict(torch.load(
    model_files[-1], map_location='cuda:0'))
generator.eval()

state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

gen = generator(latent)
