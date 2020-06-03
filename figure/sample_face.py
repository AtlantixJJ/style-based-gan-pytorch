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
parser.add_argument("--name", default="face_fewshot_full")
parser.add_argument("--n", default="1,2,4,8")
parser.add_argument("--m", default="0,1,2,3,4,5,6,7,8,9")
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
image_dir = "datasets/CelebAMask-HQ/CelebA-HQ-img"
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
examples = [int(m) for m in args.m.split(",")]

# read the datas
images = []
latents = {}
labels = []
for i in examples: # the example index
    labels.append(read_label(f"{label_dir}/{i}.png"))
    images.append(read_image(f"{image_dir}/{i}.jpg"))
    for j in inst: # svm trained instance number
        if j not in latents.keys():
            latents[j] = []
        for k in range(args.r): # the repeat index
            latent_file = f"results/face_stylegan_fewshot_real_{j}/adam_i{i}_n1000_mLL_{k:02d}_latents.npy"
            latent = torch.from_numpy(np.load(latent_file)[-1:])
            latents[j].append(latent)

# visualizations
heads = []
for idx, i in enumerate(examples):
    heads.append(images[idx])
    heads.append(colorizer(labels[idx]) / 255.)
heads = torch.cat([utils.bu(h, 256) for h in heads])
print(heads.shape)
heads = vutils.make_grid(heads,
    nrow=2, pad_value=1, padding=5).unsqueeze(0)
pads = torch.ones((1, 3, heads.shape[2], 10))

contents = []
for j in inst:
    imgs = []
    for latent in latents[j]:
        image = generator(latent.to(device))
        image = (image.detach().clamp(-1, 1) + 1) / 2
        imgs.append(image)
    imgs = torch.cat([utils.bu(i, 256).cpu() for i in imgs])
    print(imgs.shape)
    contents.append(pads)
    contents.append(vutils.make_grid(imgs,
        nrow=args.r, pad_value=1, padding=5).unsqueeze(0))
contents = torch.cat([heads] + contents, 3)

vutils.save_image(contents, f"{args.name}.png")