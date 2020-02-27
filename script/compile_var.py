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
task = sys.argv[2]

# analyze global data
files = glob.glob(f"{data_dir}/{task}*global_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in dic.keys()}
for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    for k,v in dic.items():
        v = np.array(v)
        x = v[-500:].std()
        summary[k].append(x)
utils.plot_dic(summary, "global linearity", "global.png")

# analyze classwise data
files = glob.glob(f"{data_dir}/{task}*class_dic.npy")
files.sort()
dic = np.load(files[0], allow_pickle=True)[()]
summary = {k:[] for k in utils.CELEBA_REDUCED_CATEGORY}

for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    arr = dic['IoU']
    #dic1 = {k:arr[i] for i, k in enumerate(utils.CELEBA_REDUCED_CATEGORY)}
    #utils.plot_dic(dic1, "class linearity", f.replace(".npy", ".png"))
    for i, v in enumerate(dic['IoU']):
        v = np.array(v)[-500:]

        """
        non_negative = v[v > 0]
        fluctuation = np.array([])
        if len(non_negative) > 1:
            fluctuation = np.abs(non_negative[1:] - non_negative[-1])
        negative = v[v < 0]
        negative_fluctuation = np.ones_like(negative)
        x = fluctuation.tolist() + negative_fluctuation.tolist()
        x = np.array(x).mean()
        """
        
        x = 1
        if len(v[v>0]) > 0:
            x = v.std()
        
        summary[utils.CELEBA_REDUCED_CATEGORY[i]].append(x)

utils.plot_dic(summary, "class linearity", "class.png")


