import sys
sys.path.insert(0, ".")
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import lib
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--load", default="feats.npy")
args = parser.parse_args()

# cluster
cluster_alg = lib.rcc.RccCluster()

# data
feats = np.load(args.load, allow_pickle=True)

C, H, W = feats[0].shape
X = feats.reshape(C, H * W).transpose(1, 0)
cluster_alg.fit(X)
labels, n_labels = cluster_alg.compute_assignment(1)
label_map = labels.reshape(H, W)
label_viz = utils.numpy2label(label_map, n_labels)
utils.imwrite("rcc.png", label_viz)