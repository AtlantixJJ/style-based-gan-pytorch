import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model, utils
from model.semantic_extractor import get_semantic_extractor

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def process(list):
    list = np.array(list)
    return list / list.max()

device = 'cpu'
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)
latent = torch.randn(1, 512).to(device)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]

data_dir = "record/l1"
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if os.path.isdir(f)]
model_files.sort()
model_files = [f"{m}/stylegan_linear_extractor.model"
    for m in model_files]
func = get_semantic_extractor("linear")


def concat_weight(module):
    vals = []
    ws = []
    for i, conv in enumerate(module):
        w = conv[0].weight[:, :, 0, 0]
        ws.append(w)
    ws = torch.cat(ws, 1)
    return ws


def plot_weight_layerwise(module, minimum=-1, maximum=1, subfix=""):
    for i, conv in enumerate(module):
        w = utils.torch2numpy(conv[0].weight)[:, :, 0, 0]

        fig = plt.figure(figsize=(16, 12))
        for j in range(16):
            ax = plt.subplot(4, 4, j + 1)
            ax.scatter(list(range(len(w[j]))), w[j], marker='.', s=20)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_ylim([minimum, maximum])
        plt.tight_layout()
        fig.savefig(f"l{i}{subfix}.png", bbox_inches='tight')
        plt.close()


def get_name(fp):
    ind = fp.rfind("/")
    fp = fp[ind+1:]
    ind = fp.find("l1")
    fp = fp[ind+2:]
    ind = fp.find("_")
    return fp[:ind]

def calc_subset(dic):
    indice = {}
    indice["face"] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]
    # eye glasses, ear ring, neck lace, hat, cloth
    indice["other"] = [3, 11, 12, 13, 15]
    for metric in ["IoU"]:
        for t in ["face", "other"]:
            arr = np.array(dic[metric])
            v = arr[indice[t]]
            v = v[v>-1]
            dic[f"m{metric}_{t}"] = v.mean()
    return dic


def small_absolute(x, margin=0.05):
    x[(x<margin)&(x>-margin)]=0
    return x


def surgery(state_dict, margin):
    for k in state_dict.keys():
        state_dict[k] = small_absolute(
            state_dict[k], margin)


s = []
for model_file in model_files:
    sep_model = func(n_class=16, dims=dims)
    threshold = model_file.replace(
        "/stylegan_linear_extractor.model",
        "threshold.txt")
    try:
        threshold = float(open(threshold, "r").read().strip())
    except:
        continue
    state_dict = torch.load(model_file)
    sep_model.load_state_dict(state_dict)
    w = concat_weight(sep_model.semantic_extractor)
    l1_sparsity = w.abs().sum() / w.shape[0]
    l0_sparsity = (w.abs() > threshold).sum() / np.prod(w.shape)

    file_name = model_file.replace(
        "/stylegan_linear_extractor.model",
        "_agreement.npy")
    dic = np.load(file_name, allow_pickle=True)[()]
    mIoU = dic['mIoU']
    dic["IoU"][13] = -1
    arr = np.array(dic["IoU"])
    mIoU = arr[arr>-1].mean()
    dic = calc_subset(dic)
    mIoU_face = dic['mIoU_face']
    mIoU_other = dic['mIoU_other']
    s.append("%s,%f,%f,%f,%f,%f" % (get_name(file_name),l0_sparsity, l1_sparsity, mIoU, mIoU_face, mIoU_other))
with open("sparsity.csv", "w") as f:
    f.write("\n".join(s))

lr = []
sparsity = []
miou = []
for line in s:
    items = [float(i.strip()) for i in line.split(",")]
    lr.append(items[0])
    sparsity.append(items[1])
    miou.append(items[3])
data = list(zip(sparsity, lr, miou))
data.sort(key=lambda x : x[0])
data = np.array(data)
sparsity = data[:, 0]
lr = data[:, 1]
miou = data[:, 2] / data[:, 2].max()
plt.scatter(sparsity, miou)
plt.axvline(x=0.05, c=(0.7, 0.7, 0.8))
plt.xticks(
    ticks=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    labels=["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"])
for i in range(len(sparsity)):
    plt.annotate(lr[i], (sparsity[i], miou[i]))
plt.xlabel("L0 sparsity")
plt.ylabel("Percentage of mIoU preserved")
plt.savefig('1_100_sparsity.png', box_inches="tight")
plt.close()

name = "0.0004"
model_file = f"{data_dir}/celebahq_stylegan_linear_l1{name}/stylegan_linear_extractor.model"
state_dict = torch.load(model_file, map_location="cpu")
threshold = f"{data_dir}/celebahq_stylegan_linear_l1{name}threshold.txt"
threshold = float(open(threshold, "r").read().strip())
surgery(state_dict, threshold)
sep_model.load_state_dict(state_dict)
w = concat_weight(sep_model.semantic_extractor)
with torch.no_grad():
    minimum, maximum = w.min(), w.max()

plot_weight_layerwise(
    sep_model.semantic_extractor,
    minimum, maximum, name)

fig = plt.figure(figsize=(12, 12))

mask = w.abs() > threshold
class_belonging = mask.sum(0)
ax = plt.subplot(3, 1, 1)
ax.scatter(
    np.arange(0, class_belonging.shape[0]),
    class_belonging,
    s=2,
    marker='.')

mask = w > threshold
class_belonging = mask.sum(0)
ax = plt.subplot(3, 1, 2)
ax.scatter(
    np.arange(0, class_belonging.shape[0]),
    class_belonging,
    s=2,
    marker='.')


mask = w < -threshold
class_belonging = mask.sum(0)
ax = plt.subplot(3, 1, 3)
ax.scatter(
    np.arange(0, class_belonging.shape[0]),
    class_belonging,
    s=2,
    marker='.')

plt.savefig("class_belonging.png")
plt.close()
