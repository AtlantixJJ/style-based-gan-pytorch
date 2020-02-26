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

H, W, C = feats.shape
X = feats.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)
labels = cluster_alg.fit(X)
label_map = labels.reshape(H, W)
label_viz = utils.numpy2label(label_map, labels.max() + 1)
utils.imwrite("rcc.png", label_viz)

for epsilon in np.arange(10, 100, 10):
    labels, n_components = cluster_alg.compute_assignment(epsilon)
    label_map = labels.reshape(H, W)
    label_viz = utils.numpy2label(label_map, labels.max() + 1)
    utils.imwrite(f"rcc_{epsilon}.png", label_viz)