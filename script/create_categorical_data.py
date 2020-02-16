import sys
sys.path.insert(0, ".")
import os, glob, argparse
from os.path import join as osj
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
import utils, dataset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="../datasets/CelebAMask-HQ/")
parser.add_argument("--output", default="")
args = parser.parse_args()

ds = dataset.ImageSegmentationDataset(
    root=args.dataset,
    image_dir="CelebA-HQ-img",
    label_dir="CelebAMask-HQ-mask")

category = "hat"
category_index = utils.CELEBA_REDUCED_CATEGORY.index(category)

areas = []
for idx in range(len(ds)):
    image_file = f"{ds.image_dir}/{ds.imagefiles[idx]}"
    label_file = f"{ds.label_dir}/{ds.labelfiles[idx]}"
    label = utils.imread(label_file)
    area = (label == category_index).sum()
    areas.append(area)
