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
parser.add_argument("--dataset", default="datasets/CelebAMask-HQ/")
parser.add_argument("--output", default="datasets/CelebAMask-HQ/")
parser.add_argument("--category", default="hat")
args = parser.parse_args()

ds = dataset.ImageSegmentationDataset(
    root=args.dataset,
    image_dir="CelebA-HQ-img",
    label_dir="CelebAMask-HQ-mask")

category = args.category
output_dir = f"{args.output}/{category}"
category_index = utils.CELEBA_FULL_CATEGORY.index(category)

os.system(f"mkdir {output_dir}")
os.system(f"mkdir {output_dir}/image")
os.system(f"mkdir {output_dir}/label")

areas = []
for idx in range(len(ds)):
    label_file = f"{ds.label_dir}/{ds.labelfiles[idx]}"
    label = utils.imread(label_file)
    area = (label == category_index).sum()
    areas.append(area)
areas = np.array(areas)

minimum = areas[areas > 0].min()
# plot the distribution
plt.hist(areas, bins=100, range=[minimum, areas.max()])
plt.savefig(f"{category}_distribution.png")
plt.close()

# hat: 875; eye_g: 1390
select_areas = {"hat" : 1, "eye_g" : 1, "ear_r": 1}

indice = np.where(areas > select_areas[category])[0]
for idx in indice:
    image_file = f"{ds.image_dir}/{ds.imagefiles[idx]}"
    label_file = f"{ds.label_dir}/{ds.labelfiles[idx]}"
    tar_image_file = f"{output_dir}/image/{ds.imagefiles[idx]}"
    tar_label_file = f"{output_dir}/label/{ds.labelfiles[idx]}"
    os.system(f"cp {image_file} {tar_image_file}")
    os.system(f"cp {label_file} {tar_label_file}")
