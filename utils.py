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
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def numpy2label(label_np, n_label):
    img_t = Colorize(n_label)(torch.from_numpy(label_np))
    return img_t.numpy().transpose(1, 2, 0)


def plot_dic(dic, file):
    for k, v in dic.items():
        plt.plot(v)
    plt.legend(list(dic.keys()))
    plt.savefig(file)
    plt.close()

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


def normalize_image(img):
    img[img < -1] = -1
    img[img > 1] = 1
    return (img+1)/2


def set_lerp_val(progression, lerp_val):
    for p in progression:
        p.lerp = lerp_val


def get_generator_lr(g, lr1, lr2):
    dic = []
    for i, blk in enumerate(g.progression):
        dic.append({"params": blk.conv1.parameters(), "lr": lr1})
        dic.append({"params": blk.conv2.parameters(), "lr": lr1})
        if blk.att > 0:
            dic.append({"params": blk.attention1.parameters(), "lr": lr2})
            dic.append({"params": blk.attention2.parameters(), "lr": lr2})
    return dic


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


def get_mask(styledblocks):
    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    return masks


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


def compute_score(seg, label, n=19, map_class=[(4, 5), (6, 7), (8, 9)], ignore_class=0):
    res = []
    dt_masks = []
    gt_masks = []
    for i in range(0, n):
        dt_masks.append(seg == i)
        gt_masks.append(label == i)
    if map_class is not None:
        for ct, cf in map_class:
            dt_masks[ct] = dt_masks[ct] | dt_masks[cf]
            gt_masks[ct] = gt_masks[ct] | gt_masks[cf]
    for i in range(0, n):
        if i == ignore_class:
            continue
        score = compute_iou(gt_masks[i], dt_masks[i])
        res.append(score)
    return res


def aggregate(record):
    record["class_acc"] = [-1] * 19
    total = 0
    cnt = 0
    for i in range(1, 19):
        arr = np.array(record[i])
        arr = arr[arr > -1]
        cnt += arr.shape[0]
        total += arr.sum()
        record["class_acc"][i] = arr.mean()
    record["acc"] = total / cnt
    if "sigma" in record.keys():
        record["esd"] = np.array(record["sigma"]).mean()
    return record


def summarize(record):
    label_list = ['skin', 'nose', 'eye_g', 'eye', 'r_eye', 'brow', 'r_brow', 'ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    print("=> Total accuracy: %.3f" % record["acc"])
    print("=> Class wise accuracy:")
    for i in range(1, 19):
        if 0 <= record["class_acc"][i] and record["class_acc"][i] <= 1:
            print("=> %s:\t%.3f" % (label_list[i - 1], record["class_acc"][i]))
    if "esd" in record.keys():
        print("=> Image expected standard deviation: %.3f" % record["esd"])


#########
## Logging related functions
#########


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