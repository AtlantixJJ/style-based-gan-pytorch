"""
Combine the result of OVR SVM to form a linear classifier.
"""
import sys
sys.path.insert(0, ".")
import os, pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse, glob
from thundersvm import SVC

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, model
from model.semantic_extractor import get_semantic_extractor, get_extractor_name 


print("=> Setup generator")
extractor_path = "record/celebahq1/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos-1_l1dev-1_l1unit-1/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

torch.manual_seed(65537)
device = "cuda"
total_class = 15
layer_index = 3
train_size = 16
svm_path = f"results/svm_l{layer_index}_b{train_size}.model.npy"
latent = torch.randn(1, 512, device=device)
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
#vutils.save_image(image, "image.png")
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

colorizer = utils.Colorize(total_class)

sep_model = get_semantic_extractor(get_extractor_name(extractor_path))(
    n_class=total_class,
    dims=dims)
#sep_model.load_state_dict(torch.load(extractor_path))
sep_model.to(device).eval()

svm_model = get_semantic_extractor("linear")(
    n_class=total_class,
    dims=dims,
    use_bias=True)
svm_model.to(device).eval()

def get_feature(generator, latent, noise, layer_index):
    with torch.no_grad():
        generator.set_noise(generator.parse_noise(noise))
        image, stage = generator.get_stage(latent)
    feat = stage[layer_index].detach().cpu()
    return feat

w, b, sv, segs = np.load(svm_path, allow_pickle=True)
utils.requires_grad(svm_model, False)
for i, layer in enumerate(svm_model.semantic_extractor):
    if i == layer_index:
        w_ = torch.from_numpy(w).unsqueeze(2).unsqueeze(2)
        b_ = torch.from_numpy(b).squeeze(1)
        layer[0].weight.copy_(w_)
        layer[0].bias.copy_(b_)
    else:
        layer[0].weight.fill_(0.)
        layer[0].bias.fill_(0.)
utils.requires_grad(svm_model, True)

seg = sep_model(stage)[0]
est_seg = svm_model(stage)[0]
label, estl = [utils.bu(l, 256).argmax(1) for l in [seg, est_seg]]
labels_viz = [colorizer(l).float().unsqueeze(0) / 255.
    for l in [label, estl]]
scores = [utils.bu(seg[:, i:i+1], 256) for i in range(seg.shape[1])]
scores = [s / max(-s.min(), s.max()) for s in scores]
maps = [utils.heatmap_torch(s) for s in scores]
res = [image_small] + labels_viz + maps
vutils.save_image(torch.cat(res), "svm_raw.png", nrow=3)