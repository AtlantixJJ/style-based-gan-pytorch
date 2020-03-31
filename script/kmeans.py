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
parser.add_argument("--resume", default="")
args = parser.parse_args()

files = glob.glob(f"{args.dataset}/kmeans_feats*.npy")
files.sort()

resume_i = resume_k = -1
if len(args.resume) > 0:
    resume_i, resume_k = args.resume.split("-")
    resume_i = int(resume_i)
    resume_k = int(resume_k)
    files = files[resume_i:]

for f in files:
    feats = np.load(f, allow_pickle=True)
    ind = f.find("feats")
    ind = int(f[ind+6:ind+7])
    print(f, ind)

    N, C = feats.shape
    X = feats#.reshape(H * W, C)#.reshape(C, H * W).transpose(1, 0)

    st = 2
    if resume_k > 0:
        st = resume_k

    for norm in [False, True]:
        for K in range(st, 16):
            skm = SphericalKMeans(n_clusters=K, copy_x=True, n_jobs=32, verbose=True, max_iter=1000, normalize=norm)
            skm.fit(X)
            pickle.dump(skm, open(f"{args.dataset}/skm_norm{norm}_{ind}_{K}.pkl", 'wb'))

            #labels = skm.labels_.reshape(H, W)
            #label_viz = utils.numpy2label(labels, labels.max() + 1)
            #utils.imwrite(f"{args.dataset}/skm_norm{norm}_{ind}_{K}.png", label_viz)
