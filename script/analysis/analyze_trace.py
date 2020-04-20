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
import sys, os
sys.path.insert(0, ".")

import model, utils
from model.semantic_extractor import get_semantic_extractor

# return sorted positive and negatives
def get_index(x, threshold=0.01):
    index = np.where(np.abs(x) > threshold)[0]
    vals = x[index]
    ag = vals.argsort()[::-1]
    vals = vals[ag]
    index = index[ag]
    positives = index[vals > 0]
    negatives = index[vals < 0]
    return positives, negatives

def get_dic(w, threshold=0.01):
    vals = []
    attrs = []
    
    for i in range(w.shape[0]):
        pos, neg = get_index(w[i], threshold)
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
    cr = [set() for _ in range(15)]
    for k, vs in dic.items():
        for v in vs:
            c, p = v.split("_")
            c = int(c)
            if k not in cr[c]:
                cr[c].add(k)
    return dic, t1, t2, cr


if __name__ == "__main__":
    WINDOW_SIZE = 100
    n_class = 15
    device = "cpu"
    trace_path = sys.argv[1]

    # data
    trace = np.load(trace_path) # (N, 15, D)
    weight = trace[-1]

    st, ed = 100, 101
    maximum, minimum = trace[st:ed].max(), trace[st:ed].min()
    for i in range(st, ed):
        fig = plt.figure(figsize=(12, 12))

        for j in range(trace.shape[1]):
            ax = plt.subplot(4, 4, j + 1)
            ax.plot(trace[i, j])
            ax.axes.get_xaxis().set_visible(False)
            ax.set_ylim([minimum, maximum])
        fig.savefig(f"trace_{i:02d}.png", bbox_inches='tight')
        plt.close()

