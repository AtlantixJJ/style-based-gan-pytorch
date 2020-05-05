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
    "--train-size", default=16, type=int)
parser.add_argument(
    "--layer-index", default="4", type=str)
parser.add_argument(
    "--repeat-idx", default=0, type=int)
parser.add_argument(
    "--total-class", default=15, type=int)
parser.add_argument(
    "--single-class", default=-1, type=int)
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

# build model
print("=> Use thundersvm")
from thundersvm import OneClassSVM, SVC

print("=> Setup generator")
extractor_path = "record/celebahq1/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos-1_l1dev-1_l1unit-1/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

latent = torch.randn(1, 512, device=device)
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
vutils.save_image(image_small[:16], "image.png", nrow=4)
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

# setup test
print("=> Setup test data")
# set up input
layer_index = int(args.layer_index)
colorizer = utils.Colorize(args.total_class)


def get_feature(generator, latent, noise, layer_index):
    with torch.no_grad():
        generator.set_noise(generator.parse_noise(noise))
        image, stage = generator.get_stage(latent)
    feat = stage[layer_index].detach().cpu()
    return feat


feats = np.load("sv_feat.npy")
labels = np.load("sv_label.npy")
print(f"=> Label shape: {labels.shape}")
print(f"=> Feature for SVM shape: {feats.shape}")

coefs = []
intercepts = []
segs = []
cur = 0
Cs = [args.single_class]
if args.single_class < 0:
    Cs = list(range(args.total_class))
for C in Cs:
    labels_C = labels.copy().reshape(-1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = 0
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")
    
    #feats /= np.linalg.norm(feats, 2, 1, keepdims=True)

    svm_model = SVC(kernel="linear", verbose=True)
    svm_model.fit(feats, labels_C)
    coefs.append(svm_model.coef_)
    intercepts.append(svm_model.intercept_)


model_path = f"results/sv_l{args.layer_index}_b{args.train_size}.model"
if args.single_class > 0:
    model_path = f"results/sv_l{args.layer_index}_b{args.train_size}_c{args.single_class}.model"
coefs = np.concatenate(coefs)
intercepts = np.array(intercepts)
np.save(model_path, [coefs, intercepts])