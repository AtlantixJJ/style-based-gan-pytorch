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


def plot_dic(dic, file=None):
    n = len(dic.items())
    fig = plt.figure(figsize=(4 * 7, 3 * 2))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(2, 7, i + 1)
        if type(v[0]) is list or type(v[0]) is tuple:
            arr = np.array(v)
            ax.scatter(arr[:, 0], arr[:, 1])
        else:
            ax.plot(v)
        ax.set_title(k)
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()


data_dir = "record"
tasks = ["celeba_eyeg_wgan128", "celeba_hat_wgan128", "celeba_earr_wgan128", "wgan128"][3:]
categories = ["eye_g", "hat", "ear_r", ""][3:]
# analyze global data
for category, task in zip(categories, tasks):
    files = glob.glob(f"{data_dir}/{task}/*global_dic.npy")
    files.sort()
    if len(files) == 0:
        continue
    dic = np.load(files[0], allow_pickle=True)[()]
    summary = {k:[] for k in dic.keys()}
    for f in files:
        dic = np.load(f, allow_pickle=True)[()]
        for k,v in dic.items():
            v = np.array(v)
            v = v[len(v)//2:]
            x = 1
            if len(v[v>0]) > len(v) // 10:
                x = v.std()
            summary[k].append(x)
    utils.plot_dic(summary, "", f"{task}_global.pdf")

    # analyze classwise data
    files = glob.glob(f"{data_dir}/{task}/*class_dic.npy")
    files.sort()
    dic = np.load(files[0], allow_pickle=True)[()]
    summary = {k:[] for k in utils.CELEBA_CATEGORY}
    for ind, f in enumerate(files):
        dic = np.load(f, allow_pickle=True)[()]
        arr = dic['IoU']
        for i, v in enumerate(dic['IoU'][:15]):
            v = np.array(v)
            v = v[len(v)//2:]
            x = 1
            if len(v[v>0]) > len(v)//10:
                x = v.std()
            summary[utils.CELEBA_CATEGORY[i]].append(x)

    del summary["background"]
    plot_dic(summary, f"{task}_fullclass.pdf")
    summary = {k:v for k, v in summary.items() if k == category}
    if len(summary) > 0:
        plot_dic(summary, f"{task}_class.pdf")