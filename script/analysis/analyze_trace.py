import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import sys, os, pymp
sys.path.insert(0, ".")

import model, utils
from model.semantic_extractor import get_semantic_extractor

# return sorted positive and negatives
def get_index(x, nt, pt):
    negatives = np.where(x < nt)[0]
    positives = np.where(x > pt)[0]
    return positives, negatives


def get_dic(w):
    vals = []
    attrs = []
    
    for i in range(w.shape[0]):
        nt = -w[i][w[i] < 0].std()
        pt = w[i][w[i] > 0].std()
        pos, neg = get_index(w[i], 3 * nt, 3 * pt)
        pos = pos.tolist()
        neg = neg.tolist()
        vals.extend(pos + neg)
        attrs.extend([f"{i}_p"] * len(pos) + [f"{i}_n"] * len(neg))

    # build dic
    unique_vals = np.unique(vals)
    dic = {k:[] for k in unique_vals}
    for v, a in zip(vals, attrs):
        dic[v].append(a)

    t1 = [(k, v[0]) for k, v in dic.items() if len(v) == 1]
    t2 = [(k, v[0], v[1]) for k, v in dic.items() if len(v) == 2]
    cr = [set() for _ in range(w.shape[0])]
    cp = [set() for _ in range(w.shape[0])]
    cn = [set() for _ in range(w.shape[0])]
    for k, vs in dic.items():
        for v in vs:
            c, p = v.split("_")
            c = int(c)
            if p == 'p' and k not in cp[c]:
                cp[c].add(k)
            if p == 'n' and k not in cn[c]:
                cn[c].add(k)
            if k not in cr[c]:
                cr[c].add(k)

    return dic, cr, cp, cn

if __name__ == "__main__":
    WINDOW_SIZE = 100
    n_class = 15
    device = "cpu"
    trace_path = sys.argv[1]

    # data
    trace = np.load(trace_path) # (N, 15, D)
    if "unit" in trace_path:
        trace /= np.linalg.norm(trace, 2, 2, keepdims=True)
    weight = trace[-1]

    os.system("rm video/*.png")
    maximum, minimum = trace.max(), trace.min()
    x = np.arange(0, trace.shape[2], 1)
    with pymp.Parallel(4) as p:
        #for i in tqdm(range(trace.shape[0])):
        for i in p.xrange(trace.shape[0]):
            fig = plt.figure(figsize=(12, 12))

            for j in range(trace.shape[1]):
                ax = plt.subplot(4, 4, j + 1)
                ax.scatter(x, trace[i, j], s=2)
                ax.axes.get_xaxis().set_visible(False)
                ax.set_ylim([minimum, maximum])
            fig.savefig(f"video/{i:04d}.png", bbox_inches='tight')
            plt.close()
    os.system("ffmpeg -y -f image2 -r 12 -i video/%04d.png -pix_fmt yuv420p -b:v 16000k trace_weight.mp4")

