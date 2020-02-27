import numpy as np
from spherecluster import SphericalKMeans
import pickle
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--load", default="feats.npy")
parser.add_argument("--K", default=16, type=int)
args = parser.parse_args()

feats = np.load(args.load, allow_pickle=True)

H, W, C = feats.shape
X = feats.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)
skm = SphericalKMeans(n_clusters=args.K)
skm.fit(X)
pickle.dump(skm, open("skm.pkl", 'wb'))