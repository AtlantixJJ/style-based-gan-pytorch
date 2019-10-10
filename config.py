"""
Training options and expr dir set up.
"""

import torch
import os
import argparse
from os.path import join as osj
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import utils

NUM_WORKER = 4


def config_from_name(name):
    items = name.strip().split("_")
    dics = {}
    for i in range(len(items)):
        if "att" in items[i]:
            att = int(items[i][-1])
            att_mtd = items[i+1]
            dics.update({"att": att, "att_mtd": att_mtd})
    return dics


class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Task options
        self.parser.add_argument(
            "--debug", default=False, help="Enable debugging output")
        self.parser.add_argument(
            "--task", default="ts", help="ts (teacher student training) |")
        self.parser.add_argument(
            "--name", default="", help="Name of experiment, auto inference name if leave empty")
        # Training environment options
        self.parser.add_argument("--gpu", type=str, default="0")
        self.parser.add_argument("--seed", type=int, default=1314)
        self.parser.add_argument(
            "--expr", default="expr/", help="Experiment directory")
        # Network architecture options
        # Optimize options
        self.parser.add_argument(
            "--iter-num", default=10000, help="train total iteration")
        self.parser.add_argument("--lr", default=1e-3, help="default lr")
        self.parser.add_argument(
            "--load", default="checkpoint/stylegan-1024px-new.model", help="load weight from model")
        # Train data options
        self.parser.add_argument("--batch_size", type=int, default=8)
        # Loss options

    def parse(self):
        self.args = self.parser.parse_args()
        self.debug = self.args.debug
        self.task = self.args.task
        self.n_iter = self.args.iter_num
        self.lr = self.args.lr
        self.name = self.args.name
        self.load = self.args.load
        self.batch_size = self.args.batch_size
        self.gpu = list(range(len(self.args.gpu.split(","))))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        if len(self.gpu) == 1:
            self.device = 'cuda'
            self.device1 = 'cuda:0'
            self.device2 = 'cuda:0'
        elif len(self.gpu) > 1:
            self.device = 'cuda'
            self.device1 = 'cuda:0'
            self.device2 = 'cuda:1'
        else:
            self.device = 'cpu'

        # infer if to load
        if len(self.load) > 0:
            self.load = True
            self.load_path = self.args.load
        else:
            self.load = False

    def setup(self):
        print("=> Prepare expr directory")
        os.system("mkdir %s" % self.expr_dir)
        if self.args.seed > 0:
            print("=> Set seed to %d" % self.args.seed)
            utils.set_seed(self.args.seed)

    def print_info(self):
        print("=> Task : %s" % self.task)
        print("=> Name : %s" % self.name)
        print("=> Experiment directory %s" % self.expr_dir)
        if self.load:
            print("=> Load from %s" % self.load_path)
        else:
            print("=> Train from scratch")
        print("=> LR: %.4f" % self.lr)
        print("=> Batch size : %d " % self.batch_size)


class TSConfig(BaseConfig):
    def __init__(self):
        super(TSConfig, self).__init__()
        # Optimize options
        self.parser.add_argument(
            "--stage1-step", default=200, help="Small learning rate start up.")
        self.parser.add_argument(
            "--stage1-lr", default=2e-4, help="Start up stage lr.")
        self.parser.add_argument(
            "--stage2-step", default=1000, help="Small learning rate adaption.")
        self.parser.add_argument(
            "--stage2-lr", default=2e-4, help="Adaption stage lr.")
        # Network architecture options
        self.parser.add_argument(
            "--att", type=int, default=0, help="Attention head number, 0 for no attention.")
        self.parser.add_argument(
            "--att-mtd", type=str, default="cos", help="Attention implementation type.")
        # Loss options
        self.parser.add_argument("--ma", type=float, default=-1,
                                 help="Mask area loss")
        self.parser.add_argument(
            "--md", type=float, default=-1, help="Mask divergence loss")
        self.parser.add_argument(
            "--mv", type=float, default=-1, help="Mask value loss")

    def parse(self):
        super(TSConfig, self).parse()
        self.ma = self.args.ma
        self.md = self.args.md
        self.mv = self.args.mv
        self.att = self.args.att
        self.att_mtd = self.args.att_mtd
        self.stage1_step = self.args.stage1_step
        self.stage1_lr = self.args.stage1_lr
        self.stage2_step = self.args.stage2_step
        self.stage2_lr = self.args.stage2_lr

        self.record = {'loss': [], 'mseloss': [], 'lerp_val': []}

        # auto infer name
        name = "_"
        l = ["ts"]
        if self.att > 0:
            l += ["att%d" % self.att]
            l += [self.att_mtd]
        if self.ma > 0:
            l += ["area%.2f" % self.ma]
            self.record.update({'mask_area': []})
        if self.md > 0:
            l += ["diverge%.2f" % self.md]
            self.record.update({'mask_div': []})
        if self.mv > 0:
            l += ["value%.2f" % self.md]
            self.record.update({'mask_value': []})
        name = name.join(l)

        if len(self.name) == 0:
            self.name = name

        self.expr_dir = osj("expr", self.name)

    def print_info(self):
        super(TSConfig, self).print_info()
        if self.att > 0:
            print("=> Attention head number : %d" % self.att)
            print("=> Attention method : %s" % self.att_mtd)
        if self.ma > 0:
            print("=> Mask area loss : %.2f" % self.ma)
        if self.md > 0:
            print("=> Mask divergence loss : %.2f" % self.md)
