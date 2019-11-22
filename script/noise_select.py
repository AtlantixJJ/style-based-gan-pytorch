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
args = parser.parse_args()

# constants setup
torch.manual_seed(1)
device = 'cuda'

# build model
generator = StyledGenerator().to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

rootdir = "datasets/CelebAMask-HQ/"
dlatent = np.load(osj(rootdir, "dlatent", "1.npy"))
noise = np.load(osj(rootdir, "noise", "1.npy"), allow_pickle=True)

generator.set_noise(noise1)
img1 = generator(dlatent).clamp(-1, 1)
vutils.save_image(img1, "temp.png")