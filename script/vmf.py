import numpy as np
import sys
sys.path.insert(0, ".")
from spherecluster import VonMisesFisherMixture
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
vmf_soft = VonMisesFisherMixture(n_clusters=args.K, posterior_type='soft', n_jobs=20, copy_x=False, verbose=True, max_iter=1000)
vmf_soft.fit(X)
pickle.dump(vmf_soft, open(f"vmf_soft_{args.K}.pkl", 'wb'))

labels = vmf_soft.labels_.reshape(H, W)
if args.K <= 2:
    labels = labels.reshape(1, labels.shape[0], labels.shape[1])
    label_viz = utils.heatmap_numpy(labels.astype("uint8"))[0]
    utils.imwrite(f"vmf_soft_{args.K}.png", label_viz * 255.)
else:
    labels = labels.astype("uint8")
    label_viz = utils.numpy2label(labels, labels.max() + 1)
    utils.imwrite(f"vmf_soft_{args.K}.png", label_viz * 255.)