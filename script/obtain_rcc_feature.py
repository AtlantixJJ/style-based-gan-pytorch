"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj
from sklearn.metrics.pairwise import cosine_similarity
import evaluate, utils, config, dataset
from lib.face_parsing import unet
import model, segmenter
from lib.netdissect.segviz import segment_visualization_single
from model.semantic_extractor import get_semantic_extractor

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--seed", default="")
parser.add_argument("--name", default="feats")
args = parser.parse_args()

device = "cpu"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)

torch.manual_seed(int(args.seed))
latent_size = 512
latent = torch.randn(1, latent_size).to(device)

with torch.no_grad():
    image, stage = generator.get_stage(latent)
stage = [s for s in stage if s.shape[3] >= 16]
size = max([s.shape[2] for s in stage])
data = torch.cat([
    F.interpolate(s.cpu(), size=512, mode="bilinear")[0]
        for s in stage])
data = data.permute(1, 2, 0)
print(data.shape)
np.save(args.name, utils.torch2numpy(data))
image = (image.clamp(-1, 1) + 1) / 2
vutils.save_image(image, args.name + "_image.png")