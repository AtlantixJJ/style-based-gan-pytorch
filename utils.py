from __future__ import print_function
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
from os.path import join as osj
try:
    import pickle
except:
    import cPickle as pickle
import datetime
import time
import random
from contextlib import contextmanager
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable


class MultiGPUTensor(object):
    def __init__(self, root, n_gpu):
        self.root = root
        self.n_gpu = n_gpu
        self.create_copy()
    
    def create_copy(self):
        self.val = []
        for i in range(self.n_gpu):
            self.val.append(self.root.detach().to("cuda:" + str(i)))
    
    def sync(self):
        for i in range(self.n_gpu):
            self.val[i].copy_(self.root)


######
## Colorization
######


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        if len(size) == 2:
            h, w = size
        else:
            h, w = size[1:]
            gray_image = gray_image[0]
        color_image = torch.ByteTensor(3, h, w).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    """
    edge_num = int(np.ceil(np.power(N + , 1/3))) - 1
    cmap = np.zeros((N, 3), dtype=np.uint8)
    step_size = 255. / edge_num
    cmap[0] = (0, 0, 0)
    count = 1
    for i in range(edge_num + 1):
        for j in range(edge_num + 1):
            for k in range(edge_num + 1):
                if count >= N or (i == j and j == k):
                    continue
                cmap[count] = [int(step_size * n) for n in [i, j, k]]
                count += 1
    """

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    return label_tensor.float() / 255.


def numpy2label(label_np, n_label):
    img_t = Colorize(n_label)(torch.from_numpy(label_np))
    return img_t.numpy().transpose(1, 2, 0)


######
## Image processing (PIL & Numpy based)
######


def imread(fpath):
    with open(os.path.join(fpath), "rb") as f:
        return np.asarray(Image.open(f))


def imwrite(fpath, image):
    """
    image: np array, value range in [0, 255].
    """
    if ".jpg" in fpath or ".jpeg" in fpath:
        ext = "JPEG"
    elif ".png" in fpath:
        ext = "PNG"
    with open(os.path.join(fpath), "wb") as f:
        Image.fromarray(image.astype("uint8")).save(f, format=ext)


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img
    

def normalize_image(img):
    img[img < -1] = -1
    img[img > 1] = 1
    return (img+1)/2


def set_lerp_val(progression, lerp_val):
    for p in progression:
        p.lerp = lerp_val


def get_generator_blockconv_lr(g, lr):
    dic = []
    for i, blk in enumerate(g.progression):
        dic.append({"params": blk.conv1.parameters(), "lr": lr})
        dic.append({"params": blk.conv2.parameters(), "lr": lr})
        dic.append({"params": blk.noise1.parameters(), "lr": lr})
        dic.append({"params": blk.noise2.parameters(), "lr": lr})
    return dic


def get_generator_extractor_lr(g, lr):
    dic = []
    for i, blk in enumerate(g.progression):
        if "conv" in blk.segcfg:
            dic.append({"params": blk.extractor.parameters(), "lr": lr})
    return dic


######
## Others
######


def onehot(x, n):
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)


def onehot_logit(x, n):
    x = x.argmax(1, keepdim=True)
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def infinite_dataloader(dl, total):
	i = 0
	while True:
		for sample in dl:
			i += 1
			if i == total:
				return
			yield sample
		dl.reset()


def set_seed(seed):
    random.seed(seed)
    #print('setting random-seed to {}'.format(seed))
    np.random.seed(seed)
    #print('setting np-random-seed to {}'.format(seed))
    torch.backends.cudnn.enabled = False
    #print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
    # set seed for CPU
    torch.manual_seed(seed)
    #print('setting torch-seed to {}'.format(seed))


def torch2numpy(x):
    return x.data.cpu().numpy()


def lerp(a, b, x, y, i):
    """
    Args:
        input from [a, b], output to [x, y], current position i
    """
    return (i - a) / (b - a) * (y - x) + x


class PLComposite(object):
    """
    Piecewise linear composition.
    """

    def __init__(self, st_x=0, st_y=0):
        super(PLComposite, self).__init__()
        self.ins = [st_x]
        self.outs = [st_y]

    # px should be sorted (add in sequential order)
    def add(self, px, py):
        self.ins.append(px)
        self.outs.append(py)

    def __call__(self, x):
        for i in range(1, len(self.ins)):
            if self.ins[i-1] <= x and x <= self.ins[i]:
                return lerp(self.ins[i-1], self.ins[i], self.outs[i-1], self.outs[i], x)

#########
### Evaluation
#########


def compute_iou(a, b):
    if b.any() == False:
        return -1
    return (a & b).astype("float32").sum() / (a | b).astype("float32").sum()


def idmap(x):
    id2cid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5, 8: 6, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15}
    for fr,to in id2cid.items():
        if fr == to:
            continue
        x[x == fr] = to
    return x

class MaskCelebAEval(object):
    def __init__(self, resdic=None, map_id=True):
        self.dic = {}
        self.raw_label = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        self.dic["class"] = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        self.n_class = len(self.dic["class"])
        self.ignore_classes = [0]
        self.dic["class_result"] = [[] for i in range(self.n_class)]
        self.id_to_contiguous_id()
        self.map_id = map_id

    def id_to_contiguous_id(self):
        cnt = 0
        self.id2cid = {}
        for id, name in enumerate(self.raw_label):
            if name == "l_eye" or name == "r_eye":
                name = "eye"
            if name == "l_brow" or name == "r_brow":
                name = "brow"
            if name == "l_ear" or name == "r_ear":
                name = "ear"
            self.id2cid[id] = self.dic["class"].index(name)
            cnt += 1
    
    def idmap(self, x):
        for fr,to in self.id2cid.items():
            if fr == to:
                continue
            x[x == fr] = to
        return x

    def compute_score(self, seg, label):
        res = []
        for i in range(self.n_class):
            if i in self.ignore_classes:
                score = -1
            else:
                score = compute_iou(seg == i, label == i)
            res.append(score)
        return res

    def accumulate(self, scores):
        for i, s in enumerate(scores):
            self.dic["class_result"][i].append(s)

    def aggregate(self):
        self.dic["class_acc"] = [-1] * self.n_class
        total = 0
        cnt = 0
        for i in range(self.n_class):
            arr = np.array(self.dic["class_result"][i])
            arr = arr[arr > -1]
            if arr.shape[0] == 0:
                self.dic["class_acc"][i] = -1
                continue
            cnt += arr.shape[0]
            total += arr.sum()
            self.dic["class_acc"][i] = arr.mean()
        self.dic["acc"] = total / cnt

    def summarize(self):
        print("=> Total accuracy: %.3f" % self.dic["acc"])
        print("=> Class wise accuracy:")
        for i in range(self.n_class):
            if self.dic["class_acc"][i] < 0:
                continue
            print("=> %s:\t%.3f" % (
                self.dic["class"][i],
                self.dic["class_acc"][i]))

    def save(self, fpath):
        np.save(fpath, self.dic)

#########
## Logging related functions
#########


def plot_dic(dic, file):
    for k, v in dic.items():
        plt.plot(v)
    plt.legend(list(dic.keys()))
    plt.savefig(file)
    plt.close()


def parse_log(logfile):
    with open(logfile) as f:
        head = f.readline().strip().split(" ")
        dic = {h: [] for h in head}
        lines = f.readlines()

    for l in lines:
        items = l.strip().split(" ")
        for h, v in zip(head, items):
            dic[h].append(float(v))
    return dic


"""
Args:
    record: loss dic, must have loss component
"""
def write_log(expr_dir, record):
    with open(expr_dir + "/log.txt", "w") as f:
        for key in record.keys():
            f.write("%s " % key)
        f.write("\n")
        for i in range(len(record['loss'])):
            for key in record.keys():
                try:
                    f.write("%f " % record[key][i])
                except:
                    print("!> No enough data at %s[%d]" % (key, i))
            f.write("\n")






""" Deprecated
def permute_masks(masks):
    def permute_(t):
        tmp = t[0]
        t[:-1] = t[1:]
        t[-1] = tmp
        return t

    for m in masks:
        if m is None:
            continue
        for i in range(len(m[0])):
            m[0][i] = permute_(m[0][i])
            m[1][i] = permute_(m[1][i])


def get_masks(blocks, step=6):
    masks = []
    for i, blk in enumerate(blocks):
        if blk.att > 0 and hasattr(blk.attention1, "mask"):
            masks.append([blk.attention1.mask, blk.attention2.mask])
        else:
            masks.append(None)
    return masks


def get_segmentation(blocks, step=-1, detach=True):
    seg = []
    for i, blk in enumerate(blocks):
        if step > 0 and step != i:
            continue
        if blk.n_class > 0:
            input = blk.seg_input
            if detach:
                input = input.detach()
            net_device = next(blk.extractor.parameters()).device
            if input.device != net_device:
                input = input.cpu().to(net_device)
            seg.append(blk.extractor(input))
    return seg

def visualize_masks(masks):
    masks_ = []
    for m in masks:
        if m is not None:
            masks_.extend(m[0])
            masks_.extend(m[1])
    masks_ = torch.cat([F.interpolate(m, 128) for m in masks_], 0)
    return masks_
"""
