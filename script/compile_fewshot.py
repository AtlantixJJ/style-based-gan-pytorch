import sys
sys.path.insert(0, ".")
import copy
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

# analyze global data
files = glob.glob(f"{data_dir}/*class.npy")
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
    arr[0] = arr[13] = -1 # doesn't count background & necklace
    arr = arr[arr > -0.1]
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
    ys.append(mean_dic[b])
    cs.append(colors[2])
for b in mean_dic.keys():
    xs.append(b)
    ys.append(min_dic[b])
    cs.append(colors[0])
for b in mean_dic.keys():
    xs.append(b)
    ys.append(max_dic[b])
    cs.append(colors[1])
plt.scatter(xs, ys, c=cs)
plt.savefig("scatter.png")
plt.close()