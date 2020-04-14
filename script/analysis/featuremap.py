import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj

import model, evaluate, utils, config, dataset
from lib.face_parsing import unet
from model.semantic_extractor import get_semantic_extractor

device = "cpu"
batch_size = 1
latent_size = 512
n_class = 15
latent = torch.randn(batch_size, latent_size, device=device)
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
extractor_path = "checkpoint/stylegan_unit_extractor.model"
generator = model.load_model_from_pth_file("stylegan", model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
vutils.save_image((image + 1) / 2, "image.png")
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

origin_state_dict = torch.load(extractor_path, map_location=device)
sep_model = get_semantic_extractor("unitnorm")(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(origin_state_dict)

# random projection
w = torch.randn((sum(dims),))
wp = w.clone(); wp[wp < 0] = 0
f = 0
count = 0
for s in stage:
    for i in range(s.shape[1]):
        m = F.interpolate(s[0:1, i:i+1], size=256, mode="bilinear")
        f = f + m * w[count]
        count += 1
f = (f - f.min()) / (f.max() - f.min())
img = utils.heatmap_torch(f)
vutils.save_image(img, "random_full.png")

# score map
score = sep_model(stage)[0][0]
mini, maxi = score.min(), score.max()
imgs1 = []
imgs2 = []
for i in range(score.shape[0]):
    img = score[i].clone()
    img = (img - mini) / (maxi - mini)
    img = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    imgs1.append(img)

    img = score[i]
    img[img < 0] = 0
    img = img / maxi
    img = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    imgs2.append(img)

imgs1 = torch.cat([F.interpolate(img, size=256) for img in imgs1])
imgs2 = torch.cat([F.interpolate(img, size=256) for img in imgs2])
vutils.save_image(imgs1, "score_full.png", nrow=4)
vutils.save_image(imgs2, "score_positive.png", nrow=4)


catid = 1

# weight attribution
#w = origin_state_dict["weight"][:, :, 0, 0]
#w = F.normalize(w, 2, 1)
#aw = [w[i].argsort(descending=True) for i in range(w.shape[0])]

# contribution
for i in range(len(stage)):
    stage[i].requires_grad = True

score = []
rank = []
for catid in range(2,3):
    y = sep_model(stage)[0][0, catid, :, :].sum()
    gstage = torch.autograd.grad(y, stage)
    contribution = [stage[i] * gstage[i] for i in range(len(stage))]
    layer_contrib = torch.cat([
        c[0].sum(1).sum(1) for c in contribution]).detach()
    score.append(layer_contrib)
    rank.append(layer_contrib.argsort(descending=True))
score = torch.stack(score)
rank = torch.stack(rank)
print(score.shape, rank.shape)
catid = 0
vis_imgs = []
for i in range(10):
    idx = rank[catid, i]
    stage_ind = cumdims.searchsorted(idx + 1) - 1
    stage_idx = int(idx - cumdims[stage_ind])
    img = stage[stage_ind][:, stage_idx:stage_idx+1]
    vis_imgs.append((img, i, idx, score[catid, idx]))

maxsize = max([img[0].shape[3] for img in vis_imgs])

for img, i, idx, val in vis_imgs:
    img = (img - img.min()) / (img.max() - img.min())
    if img.shape[3] <= 256:
        img = F.interpolate(img, size=256)
    img = utils.heatmap_torch(img)
    vutils.save_image(img, f"{i:02d}_{idx:04d}_{val:.3f}.png")
