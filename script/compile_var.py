import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils

# plot fid data
dic = {}
with open("/Users/jianjinxu/temp/var_metric/wgan128/wgan128_fid.txt", "r") as f:
    dic["wgan128"] = [float(num.strip()) for num in f.readlines()]
utils.plot_dic(dic, "FID", "wgan128_FID.png")

# analyze classwise data
files = glob.glob("/Users/jianjinxu/temp/var_metric/wgan128/*class_dic.npy")
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}

for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        mean = np.abs(v[1:] - v[:-1]).mean()
        #std = v[start_iteration:].std()
        summary[k].append(mean)
# filter out outlier
for k in summary.keys():
    arr = np.array(summary[k])
    mu = arr.mean()
    std = arr.std()
    arr[(arr - mu) > std] = mu + std
    arr[(mu - arr) > std] = mu - std
    summary[k] = arr
utils.plot_dic(summary, "class linearity", "class.png")
