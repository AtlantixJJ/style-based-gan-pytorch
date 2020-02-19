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
    s = "_".join(name[ind + 1:].split("_")[1:4])
    return s

def parse_key(key):
    repeat, iteration, trainsize = key.split("_")
    return int(iteration[1:]), int(trainsize[1:])


bs = []
for f in files:
    name = get_name(f)
    bs.append(int(name.split("_")[2][1:]))
    dic = np.load(f, allow_pickle=True)[()]
    arr = np.array(dic["IoU"])
    arr[0] = arr[13] = -1 # doesn't count background & necklace
    arr = arr[arr > -0.1]
    result[name] = arr.mean()
bs = np.unique(bs)
ib_dic = {i:{b:[] for b in bs} for i in [50, 100, 150, 200]}
mean_dic = copy.deepcopy(ib_dic)
std_dic = copy.deepcopy(ib_dic)
for key in result.keys():
    i, b = parse_key(key)
    ib_dic[i][b].append(result[key])
for i in ib_dic.keys():
    for b in ib_dic[i].keys():
        ib_dic[i][b] = np.array(ib_dic[i][b])
for i in ib_dic.keys():
    for b in ib_dic[i].keys():
        mean_dic[i][b] = ib_dic[i][b].mean()
        std_dic[i][b] = ib_dic[i][b].std()
print(mean_dic)
print(std_dic)

xs = []
ys = []
cs = []
for i, iter in enumerate([50, 100, 150, 200]):
    for b in mean_dic[iter].keys():
        xs.append(b)
        ys.append(mean_dic[iter][b])
        cs.append(colors[i])
plt.scatter(xs, ys, c=cs)
plt.savefig("scatter.png")
plt.legend([50, 100, 150, 200])
plt.close()