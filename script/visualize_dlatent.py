import sys
sys.path.insert(0, ".")
import os
from os.path import join as osj
import pickle
import torch
from torchvision import utils as vutils
import numpy as np
from PIL import Image
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
import utils
import lib

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt")
parser.add_argument("--name", type=str, default="27103")
args = parser.parse_args()

# constants setup
torch.manual_seed(1)
device = 'cpu'

# build model
generator = StyledGenerator().to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

rootdir = "datasets/CelebAMask-HQ/"
dlatent = np.load(osj(rootdir, "dlatent", args.name + ".npy"))
dlatent = torch.from_numpy(dlatent).float().to(device)
noise = np.load(osj(rootdir, "noise", args.name + ".npy"), allow_pickle=True)
noise = [torch.from_numpy(n).float().to(device) for n in noise]

generator.set_noise(noise)
img = generator.g_synthesis(dlatent).clamp(-1, 1)
img = (img + 1) / 2
vutils.save_image(img, "temp.png")