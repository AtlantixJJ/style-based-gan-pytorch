import sys
sys.path.insert(0, ".")
import os, pickle, glob
import matplotlib.pyplot as plt
import numpy as np

import torch
import lib.liblinear.liblinearutil as svm
from model.semantic_extractor import get_semantic_extractor, get_extractor_name 
from weight_visualization import assign_weight

def svm_train(feats, labels, total_class):
    svm_model = svm.train(labels, feats, "-n 32 -s 2 -B -1 -q")
    n_class = svm_model.get_nr_class()
    labels = svm_model.get_labels()
    coef = np.zeros((total_class, feats.shape[1]))
    for i in range(n_class):
        coef[labels[i]] = np.array(svm_model.get_decfun(i)[0])
    dims = [512, 256, 128, 64, 32]
    sep_model = get_semantic_extractor("linear")(
        n_class=total_class,
        dims=dims)
    assign_weight(sep_model.semantic_extractor, coef)
    return sep_model, coef