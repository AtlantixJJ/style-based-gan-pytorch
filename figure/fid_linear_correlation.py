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

data_dir = "results/"
fid_dir = "record/wgan64/wgan64_fid.txt"
task = sys.argv[1] #"celeba_wgan64"
ind = fid_dir.rfind("/")
fid_name = fid_dir[ind+1:-3]

# plot fid data
fids = {}
with open(fid_dir, "r") as f:
    fids[fid_name] = [float(num.strip()) for num in f.readlines()]
utils.plot_dic(fids, "FID", f"{task}_fid.png")
y = np.log(fids[fid_name])

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

fig = plt.figure(figsize=(12, 3))
for i, metric_name in enumerate(summary.keys()):
    x = summary[metric_name]
    minimum = min(y.shape[0], len(x))
    y = y[:minimum]
    x = x[:minimum]
    ax = plt.subplot(1, 4, i + 1)
    ax.scatter(x, y)
plt.savefig(f"{task}_fid_linear_separability_correlation.png")
plt.close()