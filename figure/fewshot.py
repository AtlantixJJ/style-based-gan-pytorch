import sys
sys.path.insert(0, ".")
import copy
import numpy as np
import glob
import utils
import matplotlib.pyplot as plt

# style
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

# zoom in
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


data_dir = "record/fewshot"

# analyze global data
files = glob.glob(f"{data_dir}/linear*class.npy")
files.sort()
result = {}
result_face = {}
result_other = {}

def get_name(name):
    ind = name.rfind("/")
    s = "_".join(name[ind + 1:].split("_")[1:3])
    return s

def parse_key(key):
    repeat, trainsize = key.split("_")
    return int(trainsize[1:])


face_indice = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]
other_indice = [3, 11, 12, 13, 15]

bs = []
for f in files:
    name = get_name(f)
    bs.append(int(name.split("_")[1][1:]))
    dic = np.load(f, allow_pickle=True)[()]
    arr = np.array(dic["IoU"])
    v = arr[arr > -1]
    result[name] = v.mean()
    v = arr[face_indice]
    v = v[v>-1]
    result_face[name] = v.mean()
    v = arr[other_indice]
    v = v[v>-1]
    result_other[name] = v.mean()
bs = np.unique(bs)

def get_data_from_dic(result):
    b_dic = {b:[] for b in bs}
    mean_dic = {b:[] for b in bs}
    min_dic = {b:[] for b in bs}
    max_dic = {b:[] for b in bs}
    for key in result.keys():
        b = parse_key(key)
        b_dic[b].append(result[key])
    for b in b_dic.keys():
        b_dic[b] = np.array(b_dic[b])
    for b in b_dic.keys():
        mean_dic[b] = b_dic[b].mean()
        min_dic[b] = b_dic[b].min()
        max_dic[b] = b_dic[b].max()

    xs = list(mean_dic.keys())
    means = [mean_dic[x] for x in xs]
    mins = [min_dic[x] for x in xs]
    maxs = [max_dic[x] for x in xs]
    return xs, means, mins, maxs

dics = [result, result_face, result_other]
params = [
    [[600, 0.23, 400, 0.32], [0, 64], [0.25, 0.58]],
    [[600, 0.35, 400, 0.32], [0, 64], [0.35, 0.75]],
    [[600, 0.02, 400, 0.16], [0, 64], [0.02, 0.20]]]
fig = plt.figure(figsize=(18, 7))
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    xs, means, mins, maxs = get_data_from_dic(dics[i])
    axins_box, axins_xlim, ains_ylim = params[i]
    ax.plot(xs, means, marker=".")
    ax.fill_between(xs, mins, maxs, color=colors[0])
    ax.set_xlabel("Training Size")
    ax.set_ylabel("mIoU")

    axins = ax.inset_axes(axins_box,
        transform=ax.transData)
    small = 28
    axins.plot(xs[:small], means[:small], marker=".")
    axins.fill_between(xs[:small], mins[:small], maxs[:small], color=colors[0])
    # sub region of the original image
    axins.set_xlim(*axins_xlim) # apply the x-limits
    axins.set_ylim(*ains_ylim) # apply the y-limits
    ax.indicate_inset_zoom(axins)
plt.savefig("fewshot_result.png", box_inches="tight")
plt.close()
