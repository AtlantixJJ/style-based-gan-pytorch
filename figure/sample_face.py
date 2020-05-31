"""
Given mask sample
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import sys, os, argparse, glob
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--n", default="1,2,4,8")
parser.add_argument("--m", default=4, type=int)
parser.add_argument("--r", default=2, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import model, utils, optim
from model.semantic_extractor import get_semantic_extractor, get_extractor_name

n_class = 15
device = "cuda"
colorizer = utils.Colorize(15)
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
label_dir = "datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15"
image_dir = "datasets/CelebAMask-HQ/"
# generator
generator = model.load_model(model_path)
generator.to(device).eval()

def read_label(fp):
    x = utils.imread(fp)[:, :, 0]
    x = torch.from_numpy(x).long()
    return x.unsqueeze(0).unsqueeze(0)

def read_image(fp):
    x = utils.imread(fp)
    x = torch.from_numpy(x).float() / 255.
    return x.permute(2, 0, 1).unsqueeze(0)


inst = [int(n) for n in args.n.split(",")]

images = []
latents = []
labels = []
for i in range(args.m): # the example index
    labels.append(read_label(f"{label_dir}/{i}.png"))
    images.append(read_omage(f"{image_dir}/{i}.jpg"))
    for j in inst: # svm trained instance number
        for k in range(args.r): # the repeat index
            latent_file = f"results/face_stylegan_fewshot_real_{j}/adam_i{i}_n1000_mLL_{k:02d}_latents.npy"
            latents.append(np.load(latent_file)[-1])
latents = torch.from_numpy(np.stack(latents))

# visualizations
imgs = []
nl = len(inst) * args.r
for i in range(args.m):
    imgs.append(colorizer(labels[i]) / 255.)
    for latent in latents[nl * i: nl * (i + 1)]:
        image = generator(latent.unsqueeze(0).to(device))
        image = (image.detach().clamp(-1, 1) + 1) / 2
        imgs.append(image)
imgs = torch.cat([utils.bu(i, 256).cpu() for i in imgs])

nrow = nl + 1
ncol = args.m
vutils.save_image(imgs, "test.png", nrow=nrow, padding=5, pad_value=1.)