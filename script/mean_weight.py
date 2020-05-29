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

for i in tqdm.tqdm(range(10)):
    latent.normal_()
    with torch.no_grad():
        gen, stage = generator.get_stage(latent)
        stage = [s for i, s in enumerate(stage) if i in layers]
        seg = sep_model(stage)[0]
        stage = stage[3:8]
        feat = torch.cat([utils.bu(s, 512) for s in stage], 1)
        est_label = utils.bu(seg, feat.shape[3]).argmax(1)[0]
    print(feat.shape)
    for j in range(15):
        a = (est_label == j)
        w = a.sum()
        if a.sum() <= 0:
            continue
        v = feat[0, :, a].mean(1)
        weight[j].append(utils.torch2numpy(w))
        mean_vectors[j].append(v.detach())

sep_model = get_semantic_extractor("linear")(
    n_class=15,
    dims=[512, 256, 128, 64, 32])
weight = [[w/sum(ws) for w in ws] for ws in weight]
w = torch.stack([
    sum([v * w for v, w in zip(vs, ws)])
    for vs, ws in zip(feat, weight)])
assign_weight(sep_model.semantic_extractor, w)
sep_model.to(device).eval()