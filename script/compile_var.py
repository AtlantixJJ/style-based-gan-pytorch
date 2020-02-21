import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

# plot fid data
fids = {}
with open("/Users/jianjinxu/temp/var_metric/wgan128/wgan128_fid.txt", "r") as f:
    fids["wgan128"] = [float(num.strip()) for num in f.readlines()]
utils.plot_dic(fids, "FID", "wgan128_FID.png")

# analyze global data
files = glob.glob("/Users/jianjinxu/temp/var_metric/wgan128/*global_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}

for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        x = np.abs(v[1:]-v[:-1]).mean()
        summary[k].append(x)

utils.plot_dic(summary, "global linearity", "global.png")

y = np.log(fids["wgan128"])
x = summary["mIoU"][1:1+len(y)]
plt.scatter(x, y)
plt.savefig("correlation.png")
plt.close()

# analyze classwise data
files = glob.glob("/Users/jianjinxu/temp/var_metric/wgan128/*class_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}

for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        x = np.abs(v[1:]-v[:-1]).mean()
        #std = v[start_iteration:].std()
        summary[k].append(x)

utils.plot_dic(summary, "class linearity", "class.png")


