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
total_class = 15

def load(svm_path):
    if "%d" in svm_path:
        w = []
        for i in range(total_class):
            model = svm.load_model(svm_path % i)
            v = np.array(model.get_decfun()[0])
            if model.get_labels() == [0, 1]:
                v = -v

            #v, b = np.load(svm_path % i, allow_pickle=True)
            w.append(v)
        return np.stack(w), None
    else:
        w, b, sv, segs = np.load(svm_path, allow_pickle=True)
        return w, b

svm_path = "results/sv_linear_c%d.model"
w, b = load(svm_path)
#w = np.concatenate([np.zeros((1, w.shape[1])), w])
np.save("results/sv_linear", [w, b])
