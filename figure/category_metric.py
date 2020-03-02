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
        ax.plot(v)
        ax.set_title(k)
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()


data_dir = "record"
tasks = ["celeba_eyeg_wgan128", "celeba_hat_wgan128", "wgan128"]
categories = ["eye_g", "hat", ""]
# analyze global data
for category, task in zip(categories, tasks):
    files = glob.glob(f"{data_dir}/{task}/*global_dic.npy")
    files.sort()
    dic = np.load(files[0], allow_pickle=True)[()]
    summary = {k:[] for k in dic.keys()}
    for f in files:
        dic = np.load(f, allow_pickle=True)[()]
        for k,v in dic.items():
            v = np.array(v)
            v = v[len(v)//2:]
            x = 1
            if len(v[v>0]) > 0:
                x = v.std()
            summary[k].append(x)
    utils.plot_dic(summary, "", f"{task}_global.pdf")

    # analyze classwise data
    files = glob.glob(f"{data_dir}/{task}/*class_dic.npy")
    files.sort()
    dic = np.load(files[0], allow_pickle=True)[()]
    summary = {k:[] for k in utils.CELEBA_REDUCED_CATEGORY}

    for f in files:
        dic = np.load(f, allow_pickle=True)[()]
        arr = dic['IoU']
        for i, v in enumerate(dic['IoU']):
            v = np.array(v)
            v = v[len(v)//2:]
            x = 1
            if len(v[v>0]) > 0:
                x = v.std()
            summary[utils.CELEBA_REDUCED_CATEGORY[i]].append(x)
    del summary["background"]
    del summary["neck_l"]
    plot_dic(summary, f"{task}_fullclass.pdf")
    summary = {k:v for k, v in summary.items() if k == category}
    if category == "hat":
        summary["hat"] = summary["hat"][1:]
    if len(summary) > 0:
        utils.plot_dic(summary, "", f"{task}_class.pdf")