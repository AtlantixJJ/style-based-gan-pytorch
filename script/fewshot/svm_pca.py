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

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, model
from model.semantic_extractor import get_semantic_extractor, get_extractor_name 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--total-class", default=15, type=int)
parser.add_argument(
    "--data-dir", default="datasets/SV_full")
parser.add_argument(
    "--train-size", default=1, type=int)
parser.add_argument(
    "--pca-model", default="pca.npy")
parser.add_argument(
    "--pca-size", default=32, type=int)
args = parser.parse_args()

# build model
print("=> Use liblinear (multicore)")
import lib.liblinear.liblinearutil as svm

pca = np.load(args.pca_model)[:args.pca_size]

data_dir = args.data_dir
feat_files = glob.glob(f"{data_dir}/sv_feat*.npy")
feat_files.sort()
label_files = glob.glob(f"{data_dir}/sv_label*.npy")
label_files.sort()
feat_files = feat_files[:args.train_size]
label_files = label_files[:args.train_size]

feats = np.concatenate([
    np.matmul(np.load(f), pca.transpose())
    for f in feat_files])
labels = np.concatenate([np.load(f) for f in label_files])
print(f"=> Label shape: {labels.shape}")
print(f"=> Feature for SVM shape: {feats.shape}")

model_path = f"results/svm_pca_t{args.train_size}_p{args.pca_size}.model"

coefs = []
intercepts = []
segs = []
cur = 0
for C in range(args.total_class):
    labels_C = labels.copy().reshape(-1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = 0
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")
    
    svm_model = svm.train(labels_C, feats, "-n 32 -s 2 -B -1 -q")
    coef = np.array(svm_model.get_decfun()[0])
    if svm_model.get_labels() == [0, 1]:
        coef = -coef
    coefs.append(coef)

coefs = np.stack(coefs)
coefs = np.matmul(coefs, pca)
np.save(model_path, [coefs, 0])