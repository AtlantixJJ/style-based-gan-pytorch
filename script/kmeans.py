import numpy as np
import sys
sys.path.insert(0, ".")
from spherecluster import SphericalKMeans
import pickle
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--load", default="feats.npy")
args = parser.parse_args()

feats = np.load(args.load, allow_pickle=True)

H, W, C = feats.shape
X = feats.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)

for K in [2, 3, 4, 6, 8, 12, 16]:
    skm = SphericalKMeans(n_clusters=K, copy_x=False, n_jobs=40, verbose=True, max_iter=1000)
    skm.fit(X)
    pickle.dump(skm, open(f"skm_{K}.pkl", 'wb'))

    labels = skm.labels_.reshape(H, W)
    label_viz = utils.numpy2label(labels, labels.max() + 1)
    utils.imwrite(f"skm_{K}.png", label_viz)