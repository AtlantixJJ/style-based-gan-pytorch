import numpy as np
from spherecluster import VonMisesFisherMixture
import pickle
import argparse
import utils


K = 16

parser = argparse.ArgumentParser()
parser.add_argument("--load", default="feats.npy")
parser.add_argument("--K", default=16, type=int)
args = parser.parse_args()

feats = np.load(args.load, allow_pickle=True)

H, W, C = feats.shape
X = feats.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)
vmf_soft = VonMisesFisherMixture(n_clusters=K, posterior_type='soft', n_jobs=8, verbose=True, max_iter=1000)
vmf_soft.fit(X)
pickle.dump(vmf_soft, open("vmf_soft.pkl", 'wb'))