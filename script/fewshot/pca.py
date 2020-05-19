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
    "--data-dir", default="datasets/SV_full")
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

from sklearn.decomposition import PCA

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"

data_dir = args.data_dir
feat_files = glob.glob(f"{data_dir}/sv_feat*.npy")
feat_files.sort()
label_files = glob.glob(f"{data_dir}/sv_label*.npy")
label_files.sort()
feats = np.concatenate([np.load(f) for f in feat_files])
labels = np.concatenate([np.load(f) for f in label_files])
print(f"=> Label shape: {labels.shape}")
print(f"=> Feature for PCA shape: {feats.shape}")

model = PCA().fit(feats - feats.mean(0, keepdims=True))
np.save("pca", model.components_)