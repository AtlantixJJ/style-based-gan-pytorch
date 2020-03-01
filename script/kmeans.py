import numpy as np
import sys
sys.path.insert(0, ".")
from spherecluster import SphericalKMeans
import pickle
import argparse
import utils
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="datasets/kmeans")
args = parser.parse_args()

files = glob.glob(f"{args.dataset}/kmeans_feats*.npy")
files.sort()
for f in files:
    feats = np.load(f, allow_pickle=True)
    ind = f.find("feats")
    ind = int(f[ind+6:ind+7])
    print(f, ind)

    H, W, C = feats.shape
    X = feats.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)

    for K in range(2, 16):
        skm = SphericalKMeans(n_clusters=K, copy_x=False, n_jobs=4, verbose=True, max_iter=1000)
        skm.fit(X)
        pickle.dump(skm, open(f"{args.dataset}/skm_{ind}_{K}.pkl", 'wb'))

        labels = skm.labels_.reshape(H, W)
        label_viz = utils.numpy2label(labels, labels.max() + 1)
        utils.imwrite(f"{args.dataset}/skm_{ind}_{K}.png", label_viz)
