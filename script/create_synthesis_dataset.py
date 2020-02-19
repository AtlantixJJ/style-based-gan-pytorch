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
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument(
    "--output", default="datasets/Synthesized")
parser.add_argument(
    "--number", default=10000, type=int)
parser.add_argument(
    "--seed", default=65537, type=int) # 1314 for test
args = parser.parse_args()


device = 'cuda'

# build model
generator = StyledGenerator(semantic="conv-16-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()
state_dict = torch.load(args.external_model, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()
del state_dict

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
    image = generator(latent, seg=False)
    noise = generator.get_noise()
    label = idmap.mapid(faceparser(image.clamp(-1, 1)).argmax(1))

    utils.imwrite(label_path, utils.torch2numpy(label[0]))
    np.save(latent_path, utils.torch2numpy(latent))
    np.save(noise_path, utils.torch2numpy(noise))