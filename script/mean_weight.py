"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy, math
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
from weight_visualization import assign_weight

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from weight_visualization import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoint/celebahq_stylegan_unit_extractor.model")
parser.add_argument("--G", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


device = 'cuda' if int(args.gpu) > -1 else 'cpu'
cfg = 0
batch_size = 1
EPS = 1e-6
latent_size = 512
task = "celebahq"
colorizer = utils.Colorize(15) #label to rgb
model_path = args.G
generator = model.load_model(model_path).to(device)
model_path = "checkpoint/faceparse_unet_512.pth"
model_file = args.model
savepath = model_file.replace(".model", "")
external_model = segmenter.get_segmenter(task, model_path, device)
labels, cats = external_model.get_label_and_category_names()
category_groups = utils.get_group(labels)
category_groups_label = utils.get_group(labels, False)
n_class = category_groups[-1][1]
utils.set_seed(65537)
latent = torch.randn(1, latent_size).to(device)
noise = False
op = getattr(generator, "generate_noise", None)
if callable(op):
    noise = op(device)

with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]

latent = torch.randn(batch_size, latent_size, device=device)

def get_extractor_name(model_path):
    keywords = ["nonlinear", "linear", "spherical", "generative", "projective", "unitnorm", "unit"]
    for k in keywords:
        if k in model_path:
            return k

if noise:
    generator.set_noise(noise)

layers = list(range(9))
sep_model = 0
if "layer" in model_file:
    ind = model_file.rfind("layer") + len("layer")
    s = model_file[ind:].split("_")[0]
    if ".model" in s:
        s = s.split(".")[0]
    layers = [int(i) for i in s.split(",")]
sep_model = get_semantic_extractor(get_extractor_name(model_file))(
    n_class=n_class,
    dims=np.array(dims)[layers].tolist(),
    use_bias="bias1" in model_file)
sep_model.to(device).eval()
print("=> Load from %s" % model_file)
is_resize = "spherical" not in model_file
state_dict = torch.load(model_file, map_location='cpu')
missed = sep_model.load_state_dict(state_dict)

mean_vectors = [[] for _ in range(15)]
weight = [[] for _ in range(15)]

def sphere2cartesian(x): #x: (N, C)
    r = x[:, 0:1]
    si = torch.sin(x)
    si[:, 0] = 1
    si = torch.cumprod(si, 1)
    co = torch.cos(x)
    co[:, 0] = 1
    co = torch.roll(co, -1, 1)
    return si * co * r

def arccot(x):
    # torch.atan = np.arctan
    return math.pi / 2 - torch.atan(x)

def cartesian2sphere(x): #x: (N, C)
    # (xn, xn + xn-1, ..., xn + xn-1 + ... + x2 + x1)
    cx = EPS + torch.sqrt(torch.cumsum(torch.flip(x, [1]) ** 2, 1))
    # (xn + xn-1 +...+ x2 + x1, xn + xn-1 +...+ x2, ..., xn + xn-1, xn)
    cx = torch.flip(cx, [1])
    r = cx[:, 0:1] #(N, 1)
    phi_1_n2 = arccot(x[:, :-2] / cx[:, 1:-1]) # (N, C-2)
    phi_n1 = 2 * arccot((x[:, -2] + cx[:, -2]) / x[:, -1])
    phi_n1 = phi_n1.view(-1, 1) # (N, 1)
    return torch.cat([r, phi_1_n2, phi_n1], 1)
    

for i in tqdm.tqdm(range(3000)):
    latent.normal_()
    with torch.no_grad():
        gen, stage = generator.get_stage(latent)
        stage = [s for i, s in enumerate(stage) if i in layers]
        seg = sep_model(stage)[0]
        stage = stage[3:8]
        feat = torch.cat([utils.bu(s, 512) for s in stage], 1)
        est_label = utils.bu(seg, feat.shape[3]).argmax(1)[0]
        N, C, H, W = feat.shape
        # (10000, 2544)
        #feat = utils.torch2numpy(feat[0])
        #feat = feat.reshape(C, H * W).transpose(1, 0)
        feat = feat.view(C, H * W).permute(1, 0)
        sphere_feat = cartesian2sphere(feat).view(H, W, C)
        

    for j in range(15):
        #a = utils.torch2numpy(est_label == j).astype("bool")
        a = (est_label == j)
        w = a.sum()
        if a.sum() <= 0:
            continue
        weight[j].append(utils.torch2numpy(w))
        v = sphere_feat[a, :].mean(0)
        mean_vectors[j].append(v.detach())

sep_model = get_semantic_extractor("linear")(
    n_class=15,
    dims=[512, 256, 128, 64, 32])
weight = [[w/sum(ws) for w in ws] for ws in weight]

# (15, 2544)
w = torch.stack([
    sum([v * w for v, w in zip(vs, ws)])
    for vs, ws in zip(mean_vectors, weight)])
w[:, 0] = 1. # unit
w = sphere2cartesian(w)
print((w**2).sum(1)) #verify
assign_weight(sep_model.semantic_extractor, w)
sep_model.to(device).eval()
torch.save(sep_model.state_dict(), "record/mean_weight/meannormweight_linear_layer3,4,5,6,7.model")