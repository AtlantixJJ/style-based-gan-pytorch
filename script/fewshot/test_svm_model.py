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
import lib.liblinear.liblinearutil as svm

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
svm_path = f"results/svm_train_0_c%d_l{layer_index}_b{train_size}.model"
ds = dataset.LatentSegmentationDataset(
    latent_dir="datasets/Synthesized/latent",
    noise_dir="datasets/Synthesized/noise",
    image_dir="datasets/Synthesized/image",
    seg_dir="datasets/Synthesized/label")
dl = torch.utils.data.DataLoader(ds, batch_size=train_size, shuffle=False)
latent = torch.randn(1, 512, device=device)
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True).cpu()
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

def load(svm_path):
    if "%d" in svm_path:
        w = []
        for i in range(total_class):
            model = svm.load_model(svm_path % i)
            v = np.array(model.get_decfun()[0])
            if model.get_labels() == [0, 1]:
                v = -v
            w.append(v)
        return np.stack(w), None
    else:
        w, b, sv, segs = np.load(svm_path, allow_pickle=True)
        return w, b

w, b = load(svm_path)
w_ = torch.from_numpy(w).unsqueeze(2).unsqueeze(2).float()
b_ = None
if b: b_ = torch.from_numpy(b).squeeze(1).float()

utils.requires_grad(svm_model, False)
for i, layer in enumerate(svm_model.semantic_extractor):
    if i == layer_index:
        layer[0].weight.copy_(w_)
        if b: layer[0].bias.copy_(b_)
    else:
        layer[0].weight.fill_(0.)
        layer[0].bias.fill_(0.)
utils.requires_grad(svm_model, True)

for ind, sample in enumerate(tqdm(dl)):
    break

latents, noises, images, labels = sample
latents = latents.squeeze(1)
labels = labels[:, :, :, 0].long().unsqueeze(1)
labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels]
fpath = f"results/svm_image_{ind}_b{train_size}.png"
images = images.permute(0, 3, 1, 2).float() / 255.
small_images = utils.bu(images, 256)
vutils.save_image(small_images, fpath, nrow=4)

feats = []
for i in range(latents.shape[0]):
    feat = get_feature(
        generator,
        latents[i:i+1].to(device),
        noises[i].to(device),
        layer_index)
    feats.append(feat)
feats = torch.cat(feats)

#seg = sep_model(stage)[0]
est_seg = F.conv2d(feats, w_, b_) #svm_model(stage)[0]
est_seg = est_seg[0:1]
estl = utils.bu(est_seg, 256).argmax(1)
labels_viz = [colorizer(estl).float().unsqueeze(0) / 255.]
scores = [utils.bu(est_seg[:, i:i+1], 256) for i in range(est_seg.shape[1])]
scores = [s / max(-s.min(), s.max()) for s in scores]
maps = [utils.heatmap_torch(s) for s in scores]
res = [image_small] + labels_viz + maps
vutils.save_image(torch.cat(res), "svm_raw.png", nrow=3)