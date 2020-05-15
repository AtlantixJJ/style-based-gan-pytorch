"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--train-size", default=256, type=int)
parser.add_argument(
    "--total-class", default=15, type=int)
parser.add_argument(
    "--debug", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, model
from model.semantic_extractor import get_semantic_extractor, get_extractor_name 

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"
colorizer = utils.Colorize(args.total_class)

print("=> Setup generator")
extractor_path = "record/celebahq1/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos-1_l1dev-1_l1unit-1/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

sep_model = get_semantic_extractor(get_extractor_name(extractor_path))(
    n_class=total_class,
    dims=dims)
sep_model.load_state_dict(torch.load(extractor_path))
sep_model.to(device).eval()

latent = torch.randn(1, 512, device=device)
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

data_dir = "datasets/SV_full"
feat_files = glob.glob(f"{data_dir}/sv_feat*.npy")
feat_files.sort()
label_files = glob.glob(f"{data_dir}/sv_label*.npy")
label_files.sort()
feats = np.concatenate([np.load(f) for f in feat_files])
labels = np.concatenate([np.load(f) for f in label_files])
print(f"=> Label shape: {labels.shape}")
print(f"=> Feature for SVM shape: {feats.shape}")

print("=> Calculating solver matrix")
XtX = np.matmul(feats.transpose(1, 0), feats)
try:
    XtX_1 = np.linalg.inv(XtX)
except np.linalg.LinAlgError:
    print("=> Failed to solve inversion")
    exit(-1)
solver_mat = np.matmul(XtX_1, feats.transpose(1, 0))

def predict(stage, w):
    size = stage[7].shape[3]
    feat = torch.cat([utils.bu(s, size) for s in stage[3:8]], 1)
    feat = utils.torch2numpy(feat)[0]

coefs = []
intercepts = []
svs = []
segs = []
cur = 0
for C in range(args.total_class):
    labels_C = labels.copy().reshape(-1, 1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = 0
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")

    coef = np.matmul(solver_mat, labels_C)

    # save result
    coefs.append(coef.transpose(1, 0))
    intercepts.append(0)

    # train result
    est = np.matmul(feats, coef)
    l2 = (est - labels_C).std()
    pred = est > 0.5
    label = labels_C > 0.5
    iou = (pred & label).sum().astype("float32") / (pred | label).sum()
    print(f"=> L2 distance: {l2:.3f} \t IoU: {iou:.3f}")

    # test result
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image_small = utils.bu(image.clamp(-1, 1) + 1, 256) / 2
    label = sep_model(stage)


    pred_viz, label_viz = [colorizer(torch.from_numpy(img.reshape(N, 1, H, W))).float() / 255. for img in [pred, label]]

    res = []
    count = 0
    for i in range(8):
        res.extend([label_viz[i:i+1], pred_viz[i:i+1]])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest")
        for r in res]
    fpath = f"results/l2solver_c{C}_l{args.layer_index}_b{args.train_size}.png"
    vutils.save_image(torch.cat(res), fpath, nrow=4)

model_path = f"results/l2solver_l{args.layer_index}_b{args.train_size}.model"
coefs = np.concatenate(coefs)
np.save(model_path, [coefs, 0])