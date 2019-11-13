import sys
sys.path.insert(0, ".")
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
import numpy as np
from PIL import Image
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from utils import *
import lib
matplotlib.use("agg")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/stylegan-1024px-new.model")
args = parser.parse_args()


def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))


# constants setup
torch.manual_seed(1)
device = 'cpu'
step = 8
alpha = 1
LR = 0.1
shape = 4 * 2 ** step
# set up noise
noise = []
for i in range(step + 1):
    size = 4 * 2 ** i
    noise.append(torch.randn(1, 1, size, size, device=device))
latent = torch.randn(1, 512).to(device)

# cluster
cluster_alg = lib.rcc.RccCluster()

# build model
generator = StyledGenerator().to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

out1 = generator(latent)
feat_list = generator.g_synthesis.stage
