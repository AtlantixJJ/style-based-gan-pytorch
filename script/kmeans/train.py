import numpy as np
import sys
sys.path.insert(0, ".")
#from lib.KMeansRex import RunKMeans
from libKMCUDA import kmeans_cuda
import pickle
import argparse
import utils
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="/home/jianjin/large/multiple_kmeans")
parser.add_argument("--seed", default=1314159, type=int)
args = parser.parse_args()

files = glob.glob(f"{args.dataset}/kmeans_feats*.npy")
files.sort()

for f in files:
    feats = np.load(f, allow_pickle=True)
    ind = f.find("feats")
    ind = int(f[ind+6:ind+7])
    print(f, ind)

    N, C = feats.shape
    X = feats#.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)

    # normal euclidean
    """
    for K in range(2, 16): # K [2, 15]
        centroids, assignments, avg_distance = kmeans_cuda(
            X, K, tolerance=1e-3,
            verbosity=1, seed=args.seed, average_distance=True)
        pickle.dump([centroids, avg_distance], open(f"{args.dataset}/skm_euc_{ind}_{K}.pkl", 'wb'))
    """
        
    # arccos similarity (need to normalize)
    X /= np.linalg.norm(X, 2, 1, keepdims=True)
    for K in range(2, 16): # K [2, 15]
        centroids, assignments, avg_distance = kmeans_cuda(
            X, K, tolerance=1e-3,
            metric="cos", verbosity=1, seed=args.seed, average_distance=True)
        pickle.dump([centroids, avg_distance], open(f"{args.dataset}/skm_cos_{ind}_{K}.pkl", 'wb'))

    # dot similarity
    """
    for K in range(2, 16): # K [2, 15]
        centroids, assignments, avg_distance = kmeans_cuda(
            X, K, tolerance=1e-3,
            metric="dot", verbosity=1, seed=args.seed, average_distance=True)
        pickle.dump([centroids, avg_distance], open(f"{args.dataset}/skm_dot_{ind}_{K}.pkl", 'wb'))

    # normdot product similarity (cosine similarity)
    X /= np.linalg.norm(X, 2, 1, keepdims=True)
    for K in range(2, 16): # K [2, 15]
        centroids, assignments, avg_distance = kmeans_cuda(
            X, K, tolerance=1e-3,
            metric="normdot", verbosity=1, seed=args.seed, average_distance=True)
        pickle.dump([centroids, avg_distance], open(f"{args.dataset}/skm_normdot_{ind}_{K}.pkl", 'wb'))
    """