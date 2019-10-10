from __future__ import print_function
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


def get_generator_lr(g, lr1, lr2, STEP=6):
    dic = []
    for i, blk in enumerate(g.progression):
        if i > STEP:
            break
        dic.append({"params": blk.conv1.parameters(), "lr": lr1})
        dic.append({"params": blk.conv2.parameters(), "lr": lr1})
        if blk.att > 0:
            dic.append({"params": blk.attention1.parameters(), "lr": lr2})
            dic.append({"params": blk.attention2.parameters(), "lr": lr2})
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
                    print("!> Error at %s %d" % (key, i))
            f.write("\n")


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
