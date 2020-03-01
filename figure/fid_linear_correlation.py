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
data_dir = "record/"
task = "celeba_wgan64" if imsize == "64" else "wgan128"
fid_dir = f"record/{task}/{task}_fid.txt"
ind = fid_dir.rfind("/")
fid_name = fid_dir[ind+1:-3]
metric_names = ["pixelacc", "mIoU"]
N = len(metric_names)
# plot fid data
with open(fid_dir, "r") as f:
    fids = [float(num.strip()) for num in f.readlines()]
plt.plot(fids, marker=".")
plt.xlabel("Model Snapshot")
plt.ylabel("FID")
plt.savefig(f"wgan{imsize}_fid.png", box_inches="tight")
plt.close()

y = np.array(fids)#np.log(fids)#

# analyze global data
files = glob.glob(f"{data_dir}/{task}/*global_dic.npy")
files.sort()
print(f"{data_dir}/{task}/*global_dic.npy")
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in metric_names}
for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k in summary.keys():
        v = np.array(dic[k])
        x = v[len(v)//2:].std() #np.abs(v[1:]-v[:-1]).mean()
        #x = np.log(x)
        summary[k].append(x)

fig = plt.figure(figsize=(12, 6))
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    y = y[1:minimum]
    x = x[1:minimum]
    ax = plt.subplot(2, N + 1, i + 1)
    corref = np.corrcoef(x, y)[0, 1]
    ax.set_title(metric_name + (" R=%.3f" % corref))
    #ax.set_xlabel(metric_name)
    #ax.set_ylabel("ln FID")
    ax.scatter(x, y, s=4)
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    ax = plt.subplot(2, N + 1, i + N + 2)
    #ax.set_title(metric_name)
    #ax.set_xlabel("Model Snapshot")
    #ax.set_ylabel(metric_name)
    ax.plot(x)
ax = plt.subplot(1, N + 1, N + 1)
ax.plot(fids)
ax.set_xlabel("Model Snapshot")
ax.set_ylabel("FID")
plt.tight_layout()
plt.savefig(f"{task}_fid_linear_separability_correlation.pdf",
    box_inches="tight")
plt.close()