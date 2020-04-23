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
from model.semantic_extractor import get_semantic_extractor, get_extractor_name

from script.analysis.analyze_trace import get_dic
from weight_visualization import concat_weight, assign_weight

device = "cuda"
batch_size = 1
latent_size = 512

torch.manual_seed(3)
all_latent = torch.randn(512, 512)
latent = all_latent[0:1].to(device)
#extractor_path = "record/vbs_conti/ffhq_stylegan2_unit_layer0,1,2,3,4,5,6,7,8_vbs64/stylegan2_unit_extractor.model"
extractor_path = "record/l1/celebahq_stylegan_linear_layer0,1,2,3,4,5,6,7,8_vbs8_l10.0001/stylegan_linear_extractor.model"

model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
vutils.save_image(image, "image.png")
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

origin_state_dict = torch.load(extractor_path, map_location=device)
k = list(origin_state_dict.keys())[0]
n_class = origin_state_dict[k].shape[0]
sep_model = get_semantic_extractor(get_extractor_name(extractor_path))(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(origin_state_dict)

try:
    w = concat_weight(sep_model.semantic_extractor).detach().cpu().numpy()
except:
    w = sep_model.weight[:, :, 0, 0].detach().cpu().numpy()
dic, cr, cp, cn = get_dic(w)

seg = sep_model(stage)[0]
label = seg[0].argmax(0) #(H, W)

def mean_stage(mask, stage):
    vs = []
    for s in stage:
        s = F.interpolate(s,
            size=mask.shape[0], mode="bilinear", align_corners=True)
        v = s[0, :, mask].mean(1)
        vs.append(v)
    return torch.cat(vs)

def mean_weight(w, stage):
    ws = []
    for i in range(w.shape[0]):
        cname = utils.CELEBA_CATEGORY[i]
        mask = (label == i)
        size = mask.sum()
        if size == 0:
            ws.append(torch.zeros_like(ws[-1]))
            continue

        # take the mean
        ws.append(mean_stage(mask, stage))
    return torch.stack(ws)

def multiple_mean(ws):
    f = []
    for i in range(ws[0].shape[0]):
        weights = [x[i] for x in ws if x[i].abs().sum() < 1e-3]
        if len(weights) > 0:
            f.append(sum(weights) / len(weights))
        else:
            f.append(ws[0][i])
    return torch.stack(f)


mean_model = get_semantic_extractor("linear")(
    n_class=w.shape[0],
    dims=dims).to(device)
colorizer = utils.Colorize(w.shape[0])

# single instance
mw = mean_weight(w, stage)
assign_weight(mean_model.semantic_extractor, mw)

viz_imgs = []
for i in range(3):
    latent = all_latent[i:i+1].to(device)
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = (image.clamp(-1, 1) + 1) / 2

    seg1 = sep_model(stage)[0]
    seg2 = mean_model(stage)[0]
    label1 = seg1.argmax(1)
    label2 = seg2.argmax(1)
    label1_viz = colorizer(label1).unsqueeze(0)
    label2_viz = colorizer(label2).unsqueeze(0)

    viz_imgs.extend([image, label1_viz, label2_viz])
viz_imgs = [F.interpolate(img,
    size=256, mode="bilinear", align_corners=True) for img in viz_imgs]
vutils.save_image(torch.cat(viz_imgs), "fake_kmeans_single.png", nrow=3)
    
# multiple instances
for k in [4, 8, 16, 32]:
    print(f"=> {k} instances")
    mw = []
    for i in range(k):
        latent = all_latent[i:i+1].to(device)
        with torch.no_grad():
            image, stage = generator.get_stage(latent)
            image = (image.clamp(-1, 1) + 1) / 2
        mw.append(mean_weight(w, stage))
    mw = multiple_mean(mw)
    assign_weight(mean_model.semantic_extractor, mw)

    viz_imgs = []
    for i in range(3):
        latent = all_latent[i+k:i+k+1].to(device)
        with torch.no_grad():
            image, stage = generator.get_stage(latent)
            image = (image.clamp(-1, 1) + 1) / 2
        seg1 = sep_model(stage)[0]
        seg2 = mean_model(stage)[0]
        label1 = seg1.argmax(1)
        label2 = seg2.argmax(1)
        label1_viz = colorizer(label1).unsqueeze(0) / 255.
        label2_viz = colorizer(label2).unsqueeze(0) / 255.

        viz_imgs.extend([image, label1_viz, label2_viz])
    viz_imgs = [F.interpolate(img,
        size=256, mode="bilinear", align_corners=True) for img in viz_imgs]
    vutils.save_image(torch.cat(viz_imgs), f"fake_kmeans_{k}.png", nrow=3)