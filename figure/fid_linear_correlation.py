import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm
cmap = cm.get_cmap("plasma")

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

"""
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
"""

imsize = "128"
data_dir = "record/"
task = "celeba_wgan64" if imsize == "64" else "wgan128"
fid_dir = f"record/{task}/{task}_fid.txt"
ind = fid_dir.rfind("/")
fid_name = fid_dir[ind+1:-3]
metric_names = ["mIoU"] #["pixelacc", "mIoU"]
N = len(metric_names)
# plot fid data
with open(fid_dir, "r") as f:
    fids = [float(num.strip()) for num in f.readlines()]
plt.plot(fids, marker=".")
plt.xlabel("Model Snapshot")
plt.ylabel("FID")
plt.savefig(f"wgan{imsize}_fid.pdf", box_inches="tight")
plt.close()

y = np.array(fids)#np.log(fids)#

# analyze global data
files = glob.glob(f"{data_dir}/{task}/*global_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in metric_names}
for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k in summary.keys():
        v = np.array(dic[k])
        v = v[len(v)//2:]
        x = 1
        if len(v[v>0]) > 0:
            x = v.std()
        summary[k].append(x)

"""
fig = plt.figure(figsize=(12, 6))
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    y = y[1:minimum]
    x = x[1:minimum]
    ind = np.arange(50, 50 + len(x)).reshape(1, 1, -1)
    cs = utils.heatmap_numpy(ind/ind.max())[0, 0]
    ax = plt.subplot(2, N + 1, i + 1)
    corref = np.corrcoef(x, y)[0, 1]
    ax.set_title(metric_name + (" R=%.3f" % corref))
    ax.scatter(x, y, s=4, c=cs)
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    ax = plt.subplot(2, N + 1, i + N + 2)
    ax.plot(x)
ax = plt.subplot(1, N + 1, N + 1)
ax.plot(fids)
ax.set_xlabel("Model Snapshot")
ax.set_ylabel("FID")
"""

metric_name = metric_names[0]

x = summary[metric_name]

# ensure length equal
minimum = min(y.shape[0], len(x))
y = y[1:minimum]
x = np.array(x[1:minimum])

# colorization
ind = np.arange(3, 3 + len(x)).reshape(1, 1, -1)
cs = cmap(ind/55.)[0, 0]

fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 3, 1)
ax.plot(x)
ax.set_xlabel("Model Snapshot")
ax.set_ylabel("TSL")

ax = plt.subplot(1, 3, 2)
ax.plot(fids[1:minimum])
ax.set_xlabel("Model Snapshot")
ax.set_ylabel("FID")

corref = np.corrcoef(x, y)[0, 1]
ax = plt.subplot(1, 3, 3)
ax.set_title("R=%.3f" % corref)
ax.scatter(x, y, s=6, c=cs)
ax.set_xlabel("TSL")
ax.set_ylabel("FID") 
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(
    cm.ScalarMappable(cmap=cmap),
    cax=cax)
ticks = [0, 10, 20, 30, 40, 50]
cbar.set_ticks([t/50. for t in ticks])
cbar.set_ticklabels([str(t) for t in ticks])

for i in range(1, 5):
    ax.annotate(str(i), (x[i-1]+0.001, y[i-1]+4))
ax.annotate("5", (x[4]+0.001, y[4]-5))

plt.tight_layout()
plt.savefig(f"{task}_fid_linear_separability_correlation.pdf",
    box_inches="tight")
plt.close()