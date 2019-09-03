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
import torchvision.utils as vutils
from torch.autograd import Variable


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