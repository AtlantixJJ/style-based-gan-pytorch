"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, argparse, pickle
from tqdm import tqdm
import torch
import numpy as np
from model.tfseg import StyledGenerator
import torch.nn.functional as F
from torchvision import utils as vutils
import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--output", default="datasets/Synthesized")
parser.add_argument(
    "--number", default=100, type=int)
args = parser.parse_args()


device = 'cuda'

# build model
generator = StyledGenerator(semantic="conv-16-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

# setup
torch.manual_seed(65537)
latent = torch.randn(1, 512, device=device)
for folder in ["latent", "noise", "label"]:
    os.system(f"mkdir {args.output}/{folder}")


for ind in tqdm(range(args.number)):
    latent_path = f"{args.output}/latent/{ind:03d}.npy"
    noise_path = f"{args.output}/noise/{ind:03d}.npy"
    label_path = f"{args.output}/label/{ind:03d}.png"
    latent.normal_()
    image, seg = generator(latent)
    noise = generator.get_noise()
    label = seg.argmax(1)
    utils.imwrite(label_path, utils.torch2numpy(label[0]))
    np.save(latent_path, utils.torch2numpy(latent))
    np.save(noise_path, utils.torch2numpy(noise))