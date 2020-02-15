"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from torchvision import utils as vutils
import utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--output", default="datasets/Synthesized")
parser.add_argument(
    "--number", default=1, type=int)
args = parser.parse_args()

# constants setup
torch.manual_seed(65537)
device = 'gpu'

# build model
generator = StyledGenerator(semantic="conv-16-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

latent = torch.randn(1, 512, device=device)

for ind in range(args.number):
    latent.normal_()
    image, seg = generator(latent)
    label = seg.argmax(1)
    