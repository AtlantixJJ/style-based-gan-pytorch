"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, argparse, pickle
from tqdm import tqdm
import torch
import numpy as np
from model.tf import StyledGenerator
import torch.nn.functional as F
from torchvision import utils as vutils
import utils
from segmenter import get_segmenter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument(
    "--output", default="datasets/Synthesized")
parser.add_argument(
    "--number", default=1000, type=int)
parser.add_argument(
    "--seed", default=65537, type=int) # 1314 for test
args = parser.parse_args()


device = 'cuda'

# build model
generator = StyledGenerator().to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()
external_model = get_segmenter("celebahq", args.external_model)

# setup
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
for folder in ["latent", "noise", "label"]:
    os.system(f"mkdir {args.output}/{folder}")

idmap = utils.CelebAIDMap()

for ind in tqdm(range(args.number)):
    latent_path = f"{args.output}/latent/{ind:05d}.npy"
    noise_path = f"{args.output}/noise/{ind:05d}.npy"
    label_path = f"{args.output}/label/{ind:05d}.png"
    latent.normal_()
    with torch.no_grad():
        image = generator(latent)
        noise = generator.get_noise()
    label = external_model.segment_batch(image.clamp(-1, 1)).argmax(1)

    utils.imwrite(label_path, utils.torch2numpy(label[0]))
    np.save(latent_path, utils.torch2numpy(latent))
    np.save(noise_path, utils.torch2numpy(noise))