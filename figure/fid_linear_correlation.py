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

imsize = sys.argv[1]
data_dir = "results/"
fid_dir = f"record/wgan{imsize}/wgan{imsize}_fid.txt"
task = "celeba_wgan64" if imsize == "64" else "wgan128"
ind = fid_dir.rfind("/")
fid_name = fid_dir[ind+1:-3]

# plot fid data
with open(fid_dir, "r") as f:
    fids = [float(num.strip()) for num in f.readlines()]
plt.plot(fids, marker=".")
plt.xlabel("Model Snapshot")
plt.ylabel("FID")
plt.savefig(f"wgan{imsize}_fid.png", box_inches="tight")
plt.close()

y = np.log(fids)

# analyze global data
files = glob.glob(f"{data_dir}/{task}*global_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}
for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        x = np.abs(v[1:]-v[:-1]).mean()
        summary[k].append(x)

fig = plt.figure(figsize=(12, 6))
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    y = y[1:minimum]
    x = x[1:minimum]
    ax = plt.subplot(2, 5, i + 1)
    ax.set_title(metric_name + " correlation")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("ln FID")
    ax.scatter(x, y)
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    ax = plt.subplot(2, 5, i + 6)
    ax.set_title(metric_name)
    ax.set_xlabel("Model Snapshot")
    ax.set_ylabel(metric_name)
    ax.plot(x)
ax = plt.subplot(1, 5, 5)
ax.plot(fids)
ax.set_xlabel("Model Snapshot")
ax.set_ylabel("FID")
plt.savefig(f"{task}_fid_linear_separability_correlation.png",
    box_inches="tight")
plt.close()