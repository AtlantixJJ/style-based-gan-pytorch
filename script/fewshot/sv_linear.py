"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, pickle, glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--name", default="stylegan")
parser.add_argument(
    "--data-dir", default="datasets/SV_full")
parser.add_argument(
    "--train-size", default=1, type=int)
parser.add_argument(
    "--gpu", default="0")
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
from weight_visualization import assign_weight

USE_THUNDER = False

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"

# build model
print("=> Use liblinear (multicore)")
import lib.liblinear.liblinearutil as svm

data_dir = args.data_dir
feat_files = glob.glob(f"{data_dir}/sv_feat*.npy")
feat_files.sort()
label_files = glob.glob(f"{data_dir}/sv_label*.npy")
label_files.sort()
feat_files = feat_files[:args.train_size]
label_files = label_files[:args.train_size]
feats = np.concatenate([np.load(f) for f in feat_files])
labels = np.concatenate([np.load(f) for f in label_files])
#idx = np.random.choice(range(feats.shape[0]), 2000, replace=False)
#feats = feats[idx]
#labels = labels[idx]
print(f"=> Label shape: {labels.shape}")
print(f"=> Feature for SVM shape: {feats.shape}")

subfix = ""
model_path = f"results/sv_linear_t{args.train_size}"
model_path += "{subfix}.model"

coefs = []
intercepts = []
segs = []
cur = 0
Cs = [args.single_class]
if args.single_class < 0:
    Cs = list(range(args.total_class))

""" # manual one v.s. rest
for C in Cs:
    labels_C = labels.copy().reshape(-1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = 0
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")
    
    svm_model = svm.train(labels_C, feats, "-n 32 -s 2 -B -1 -q")
    if args.single_class > -1:
        subfix = f"_c{args.single_class}"
    svm.save_model(model_path.format(subfix=subfix), svm_model)
    coef = np.array(svm_model.get_decfun()[0])
    coefs.append(coef)
    intercepts.append(0)

if args.single_class > -1:
    subfix = f"_c{args.single_class}"
coefs = np.concatenate(coefs)
intercepts = np.array(intercepts)
np.save(model_path.format(subfix=subfix), [coefs, intercepts])
"""

# liblinear multiclass one v.s. rest
svm_model = svm.train(labels, feats, "-n 32 -s 2 -B -1 -q")
est_labels, acc, vals = svm.predict(labels, feats, svm_model)
svm.save_model(
    model_path.format(subfix=subfix).replace(".model", ".ll"),
    svm_model)
n_class = svm_model.get_nr_class()
labels = svm_model.get_labels()
coef = np.zeros((args.total_class, feats.shape[1]))
for i in range(n_class):
    coef[labels[i]] = np.array(svm_model.get_decfun(i)[0])

layers, dims = open(f"{args.data_dir}/dims.txt", "r").readlines()
dims = [int(d) for d in dims.strip().split(" ")]
layer = layers.strip().replace(" ", ",")

sep_model = get_semantic_extractor("linear")(
    n_class=args.total_class,
    dims=dims).to(device)
assign_weight(sep_model.semantic_extractor, coef)
print(f"=> Saved to results/fewshot_svm/svm_t{args.train_size}_{args.name}_layer{layer}_linear_extractor.model")
torch.save(
    sep_model.state_dict(),
    f"results/fewshot_svm/svm_t{args.train_size}_{args.name}_layer{layer}_linear_extractor.model")