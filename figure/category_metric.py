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

data_dir = "record"
tasks = ["celebahq_eyeg_wgan128", "celebahq_hat_wgan128", "celebahq_wgan128"]

# analyze global data
for task in tasks:
    files = glob.glob(f"{data_dir}/{task}/*global_dic.npy")
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
    files = glob.glob(f"{data_dir}/{task}/*class_dic.npy")
    files.sort()
    dic = np.load(files[0], allow_pickle=True)[()]
    summary = {k:[] for k in utils.CELEBA_REDUCED_CATEGORY}

    for f in files:
        dic = np.load(f, allow_pickle=True)[()]
        arr = dic['IoU']
        for i, v in enumerate(dic['IoU']):
            v = np.array(v)[-500:]
            x = 1
            if len(v[v>0]) > 0:
                x = v.std()
            summary[utils.CELEBA_REDUCED_CATEGORY[i]].append(x)
    utils.plot_dic(summary, "class linearity", f"{task}_fullclass.pdf")
    summary = {k:v for k, v in summary.items() if k in task}
    if len(summary) > 0:
        utils.plot_dic(summary, "class linearity", f"{task}_class.pdf")