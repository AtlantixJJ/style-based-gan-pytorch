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

USE_THUNDER = False

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"

# build model
if USE_THUNDER:
    print("=> Use thundersvm")
    from thundersvm import OneClassSVM, SVC
else:
    print("=> Use liblinear (multicore)")
    import lib.liblinear.liblinearutil as svm

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
    if USE_THUNDER:
        svm_model = SVC(kernel="linear", verbose=True)
        svm_model.fit(feats, labels_C)
        coefs.append(svm_model.coef_)
        intercepts.append(svm_model.intercept_)
    else:
        svm_model = svm.train(labels_C, feats, "-n 32 -s 2 -B -1 -q")
        #if args.single_class > 0:
        #    model_path = f"results/sv_linear_c{args.single_class}.model"
        #svm.save_model(model_path, svm_model)
        coef = np.array(svm_model.get_decfun()[0])
        coefs.append(coef)
        intercepts.append(0)


model_path = f"results/sv.model"
if args.single_class > -1:
    model_path = f"results/sv_c{args.single_class}.model"
if not USE_THUNDER:
    model_path = model_path.replace("sv", "sv_liblinear")
coefs = np.concatenate(coefs)
intercepts = np.array(intercepts)
np.save(model_path, [coefs, intercepts])