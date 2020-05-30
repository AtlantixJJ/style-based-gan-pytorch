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
from weight_visualization import plot_weight_concat

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="results/l2solver_l4_b245.model.npy")
args = parser.parse_args()
name = args.model[args.model.rfind("/") + 1:args.model.find(".")]

print("=> Setup generator")
extractor_path = "record/celebahq1/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos-1_l1dev-1_l1unit-1/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

torch.manual_seed(65537)
device = "cuda"
total_class = 15
layer_index = 3
train_size = 16
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
sep_model.load_state_dict(torch.load(extractor_path))
sep_model.to(device).eval()

svm_model = get_semantic_extractor("linear")(
    n_class=total_class,
    dims=dims,
    use_bias=True)
svm_model.to(device).eval()


latent = torch.randn(8, 512, device=device)


def evaluate_multiple_layer():
    w = np.load(args.model, allow_pickle=True)
    w = w[:36]
    w_ = torch.from_numpy(w).float().unsqueeze(2).unsqueeze(2)

    res = []
    #minimum, maximum = w.min(), w.max()
    plot_weight_concat(w[:16], -1, 1, f"{name}_pca_weight.png")

    for i in range(latent.shape[0]):
        with torch.no_grad():
            images, stage = generator.get_stage(latent[i:i+1])
        images = utils.bu((images.clamp(-1, 1) + 1) / 2, 256).cpu()
        feats = stage[3:8]
        maxsize = max([s.shape[3] for s in feats])
        feats = torch.cat([utils.bu(s, maxsize) for s in feats], 1)
        feats = feats.detach().cpu()

        seg = sep_model(stage)[0].cpu()
        label = utils.bu(seg, 256).argmax(1)
        label_viz = colorizer(label).float() / 255.
        est_seg = F.conv2d(feats, w_)
        estl = utils.bu(est_seg, 256).argmax(1)
        estl_viz = colorizer(estl).float() / 255.

        res.extend([images, label_viz, estl_viz])
    res = torch.cat(res)
    #vutils.save_image(res, f"{name}_pca_raw_all.png", nrow=3)

    scores = [utils.bu(est_seg[0:1, i:i+1], 256) for i in range(est_seg.shape[1])]
    scores = [s / max(-s.min(), s.max()) for s in scores]
    maps = [utils.heatmap_torch(s) for s in scores]
    res = [images, label_viz, estl_viz] + maps
    vutils.save_image(torch.cat(res), f"{name}_pca_raw_score.png", nrow=6)

evaluate_multiple_layer()