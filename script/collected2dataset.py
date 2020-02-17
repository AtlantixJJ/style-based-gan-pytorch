"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, argparse, pickle, glob
from tqdm import tqdm
import torch
import numpy as np
from model.tfseg import StyledGenerator
import torch.nn.functional as F
from torchvision import utils as vutils
import utils


LABEL_COLORS = [
    [0, 0, 0,      ],
    [208, 2, 27,   ],
    [245, 166, 35, ],
    [248, 231, 28, ],
    [139, 87, 42,  ],
    [126, 211, 33, ],
    [255, 255, 255 ],
    [226, 238, 244,],
    [226, 178, 213,],
    [189, 16, 224, ],
    [74, 144, 226, ],
    [80, 227, 194, ]]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", default="data/bedroom_coarse")
parser.add_argument(
    "--output", default="datasets/Bedroom_label_coarse")
args = parser.parse_args()


if not os.path.exists(args.output):
    os.mkdir(args.output)


for folder in ["latent", "noise", "label"]:
    if not os.path.exists(f"{args.output}/{folder}"):
        os.mkdir(f"{args.output}/{folder}")


# first: convert label stroke to labels
label_files = glob.glob(f"{args.data_dir}/*image_stroke.png")
latent_files = glob.glob(f"{args.data_dir}/*origin_latent.npy")
noise_files = glob.glob(f"{args.data_dir}/*origin_noise.npy")
label_files.sort()
latent_files.sort()
noise_files.sort()

for i in range(len(label_files)):
    label = torch.from_numpy(utils.imread(label_files[i])).permute(2, 0, 1)
    label = utils.rgb2label(label, LABEL_COLORS)
    utils.imwrite(f"{args.output}/label/{i:03d}.png", utils.torch2numpy(label))
    os.system(f"cp {latent_files[i]} {args.output}/latent/{i:03d}.npy")
    os.system(f"cp {noise_files[i]} {args.output}/noise/{i:03d}.npy")

