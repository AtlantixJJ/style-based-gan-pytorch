import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils

# analyze classwise data
files = glob.glob("results/var_metric/*class*.npy")
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}

for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        mean = v.mean()
        std = v.std()
        if mean < 0:
            std = -1
        summary[k].append(std)
utils.plot_dic(summary, "class linearity", "class.png")
