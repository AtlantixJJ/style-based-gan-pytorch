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

data_dir = sys.argv[1]
fid_dir = sys.argv[2]
ind = fid_dir.rfind("/")
fid_name = fid_dir[ind+1:-3]
# plot fid data
fids = {}
with open(fid_dir, "r") as f:
    fids[fid_name] = [float(num.strip()) for num in f.readlines()]
utils.plot_dic(fids, "FID", f"{fid_name}.png")

# analyze global data
files = glob.glob(f"{data_dir}/*global_dic.npy")
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

y = np.log(fids[fid_name])
x = summary["mIoU"][1:1+len(y)]
plt.scatter(x, y)
plt.savefig("correlation.png")
plt.close()

# analyze classwise data
files = glob.glob(f"{data_dir}/*class_dic.npy")
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


