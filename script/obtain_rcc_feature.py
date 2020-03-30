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
#parser.add_argument("--seed", default="")
parser.add_argument("--name", default="feats")
args = parser.parse_args()

device = "cpu"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)

#torch.manual_seed(int(args.seed))
latent_size = 512
maxsize = 512
N = 16
latent = torch.randn(N, latent_size).to(device)
bundle = []
images = []

for i in range(N):
    with torch.no_grad():
        image, stage = generator.get_stage(latent[i:i+1])
        images.append(image)
        size = max([s.shape[2] for s in stage])
        data = torch.cat([
            F.interpolate(s.cpu(), size=maxsize, mode="bilinear")[0]
                for s in stage]) # (C, H, W)
        ind = torch.randperm(data.shape[1] * data.shape[2])
        ind = ind[:len(ind) // N]
        data = data.view(data.shape[0], -1)[:, ind] # (C, N)
        bundle.append(utils.torch2numpy(data.permute(1, 0))) # (N, C)

np.save(args.name, np.concatenate(bundle))
image = (image.clamp(-1, 1) + 1) / 2
vutils.save_image(torch.cat(images), args.name + "_image.png")