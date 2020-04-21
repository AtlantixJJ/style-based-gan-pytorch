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

from script.analysis.analyze_trace import get_dic
from weight_visualization import concat_weight

device = "cuda"
batch_size = 1
latent_size = 512
n_class = 15
latent = torch.randn(batch_size, latent_size, device=device)
#model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
model_path = "checkpoint/face_ffhq_1024x1024_stylegan2.pth"
extractor_path = "record/vbs/ffhq_stylegan2_linear_layer0,1,2,3,4,5,6,7,8_vbs4/stylegan2_linear_extractor.model"
generator = model.load_stylegan2(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image, size=256, mode="bilinear", align_corners=True)
vutils.save_image(image, "image.png")
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

origin_state_dict = torch.load(extractor_path, map_location=device)
sep_model = get_semantic_extractor("linear")(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(origin_state_dict)

w = concat_weight(sep_model.semantic_extractor).detach().cpu().numpy()
dic, cr, cp, cn = get_dic(w, 0.1)

# Selected featuremaps
"""
print("=> Visualize selected featuremaps")
C = 1 # face
indice = cr[C]
viz_imgs = []
for idx in indice:
    stage_ind = cumdims.searchsorted(idx + 1) - 1
    stage_idx = int(idx - cumdims[stage_ind])
    img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
    attr = "-".join(dic[idx])
    viz_imgs.append((img, idx, attr))

for i, (img, idx, attr) in enumerate(viz_imgs):
    if img.shape[3] >= 1024:
        continue
    img = img / img.max() # the featuremap is positive
    if img.shape[3] <= 256:
        img = F.interpolate(img, size=256, mode="bilinear", align_corners=True)
    img = utils.heatmap_torch(img)
    vutils.save_image(img, f"{i:03d}_{idx:04d}_{attr}.png")
"""

"""
print("=> Show random subsamples")
C = 2 # nose
indice = list(cr[C])
cname = utils.CELEBA_CATEGORY[C]
print(f"=> Category {cname} size {len(indice)}")
count = 0
print(indice)
for k in [10, 40, 100, 200, len(indice)]:
    repeat_num = 1 if k == len(indice) else 5
    for i in range(repeat_num): # repeat
        sample = np.random.choice(indice, size=k, replace=False)
        s = []
        for idx in sample:
            stage_ind = cumdims.searchsorted(idx + 1) - 1
            stage_idx = int(idx - cumdims[stage_ind])
            img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
            s.append(img * w[C, idx])
        size = max([a.shape[2] for a in s])
        img = sum([F.interpolate(a, size=size, mode="bilinear", align_corners=True)
            for a in s])
        img = utils.heatmap_torch(img / img.max())
vutils.save_image(img, f"{cname}_sample.png")
count += 1
"""

def partial_sum(w, indice, stage, size=256):
    s = []
    for idx in indice:
        stage_ind = cumdims.searchsorted(idx + 1) - 1
        stage_idx = int(idx - cumdims[stage_ind])
        img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
        s.append(img * w[idx])
    return sum([F.interpolate(a, size=size, mode="bilinear", align_corners=True)
        for a in s])

C = 1 # skin
print("=> Positive and negative")
indice = list(cr[C])
cname = utils.CELEBA_CATEGORY[C]
print(f"=> Category {cname} size {len(indice)}")

for i in range(3): # repeat
    latent.normal_()
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = (image.clamp(-1, 1) + 1) / 2
        image_small = F.interpolate(image, size=256, mode="bilinear", align_corners=True)

    #msp = partial_sum(w[C], list(cp[C]), stage)
    #msn = -partial_sum(w[C], list(cn[C]), stage)
    #ms = partial_sum(w[C], list(cr[C]), stage)

    mp = partial_sum(w[C], np.where(w[C] > 0)[0], stage)
    mn = -partial_sum(w[C], np.where(w[C] < 0)[0], stage)
    mt = partial_sum(w[C], np.where(w[C] != 0)[0], stage)

    sp = partial_sum(w[C], np.where((0 <= w[C]) & (w[C] < 0.1))[0], stage)
    sn = -partial_sum(w[C], np.where((-0.1 <= w[C]) & (w[C] < 0))[0], stage)
    lp = partial_sum(w[C], np.where(0.1 <= w[C])[0], stage)
    ln = -partial_sum(w[C], np.where(w[C] < -0.1)[0], stage)
    diff_s = sp - sn
    diff_l = lp - ln
    res = sp - sn + lp - ln
    arr = [mp, mn, mt, sp, sn, lp, ln, diff_s, diff_l, res]
    maxi = max([a.max() for a in arr])

    img_p, img_n, img_t, img_sp, img_sn, img_lp, img_ln, imgdiff_s, imgdiff_l, imgres = [
        utils.heatmap_torch(m / maxi) for m in arr]
    
    img = torch.cat([
        img_p, img_n, image_small, img_t,
        img_sp, img_sn, imgdiff_s, imgres,
        img_lp, img_ln, imgdiff_l])
    vutils.save_image(img, f"{i}_{cname}_positive_negative.png", nrow=4)


# random projection
print("=> Random projection")
w = torch.randn((sum(dims),))
wp = w.clone(); wp[wp < 0] = 0
f = 0
count = 0
for s in stage:
    for i in range(s.shape[1]):
        m = F.interpolate(s[0:1, i:i+1], size=256, mode="bilinear", align_corners=True)
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
"""
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
    img = img / img.max() # the featuremap is positive
    if img.shape[3] <= 256:
        img = F.interpolate(img, size=256)
    img = utils.heatmap_torch(img)
    vutils.save_image(img, f"{i:02d}_{idx:04d}_{val:.3f}.png")
"""