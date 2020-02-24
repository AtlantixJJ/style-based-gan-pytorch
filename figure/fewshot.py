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


data_dir = "results/"

# analyze global data
files = glob.glob(f"{data_dir}/linear*class.npy")
files.sort()
result = {}

def get_name(name):
    ind = name.rfind("/")
    s = "_".join(name[ind + 1:].split("_")[1:3])
    return s

def parse_key(key):
    repeat, trainsize = key.split("_")
    return int(trainsize[1:])


bs = []
for f in files:
    name = get_name(f)
    bs.append(int(name.split("_")[1][1:]))
    dic = np.load(f, allow_pickle=True)[()]
    arr = np.array(dic["IoU"])
    arr = arr[arr > -1]
    result[name] = arr.mean()
bs = np.unique(bs)
b_dic = {b:[] for b in bs}
mean_dic = copy.deepcopy(b_dic)
min_dic = copy.deepcopy(b_dic)
max_dic = copy.deepcopy(b_dic)
for key in result.keys():
    b = parse_key(key)
    b_dic[b].append(result[key])
for b in b_dic.keys():
    b_dic[b] = np.array(b_dic[b])
for b in b_dic.keys():
    mean_dic[b] = b_dic[b].mean()
    min_dic[b] = b_dic[b].min()
    max_dic[b] = b_dic[b].max()

xs = []
ys = []
cs = []
for b in mean_dic.keys():
    xs.append(b)
for b in mean_dic.keys():
    ys.append(min_dic[b])
for b in mean_dic.keys():
    ys.append(max_dic[b])

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(xs, [mean_dic[x] for x in xs], marker=".")
ax.fill_between(xs, [min_dic[x] for x in xs], [max_dic[x] for x in xs], color=colors[0])
ax.set_xlabel("Training Size")
ax.set_ylabel("mIoU")

axins = ax.inset_axes([500, 0.3, 1, 1])
small = 15
axins.plot(
    xs[:small],
    [mean_dic[x] for x in xs][:small], marker=".")
axins.fill_between(
    xs[:small],
    [min_dic[x] for x in xs][:small],
    [max_dic[x] for x in xs][:small], color=colors[0])
# sub region of the original image
axins.set_xlim(0, 30) # apply the x-limits
axins.set_ylim(0.25, 0.6) # apply the y-limits
ax.indicate_inset_zoom(axins)
#axins.set_xticklabels('')
#axins.set_yticklabels('')

"""
small = 15
axins = zoomed_inset_axes(ax, 1, loc=4) # zoom-factor: 2.5, location: upper-left
axins.plot(
    xs[:small],
    [mean_dic[x] for x in xs][:small], marker=".")
axins.fill_between(
    xs[:small],
    [min_dic[x] for x in xs][:small],
    [max_dic[x] for x in xs][:small], color=colors[0])
axins.set_xlim(0, 30) # apply the x-limits
axins.set_ylim(0.25, 0.6) # apply the y-limits
"""

plt.savefig("fewshot_result.png", box_inches="tight")
plt.close()
