"""
Training options and expr dir set up.
"""

import torch
import os
import argparse
from os.path import join as osj
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import dataset
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
        if "seg" in items[i]:
            segcfg = items[i+2]
            dics.update({"semantic": segcfg})
    return dics


class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Task options
        self.parser.add_argument(
            "--debug", default=False, help="Enable debugging output")
        self.parser.add_argument(
            "--task", default="ts", help="ts (teacher student training) | seg (given segmentation label)")
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
            "--iter-num", default=10000, type=int, help="train total iteration")
        self.parser.add_argument("--lr", default=1e-3, type=float, help="default lr")
        self.parser.add_argument(
            "--load", default="checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt", help="load weight from model")
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
        self.n_gpu = len(self.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        if len(self.gpu) == 1:
            self.device = 'cuda'
            self.device1 = 'cuda:0'
            self.device2 = 'cuda:0'
            self.device3 = 'cuda:0'
        elif len(self.gpu) > 1:
            self.device = 'cuda'
            self.device1 = 'cuda:0'
            self.device2 = 'cuda:1'
            self.device3 = 'cuda:2'
        else:
            self.device = 'cpu'

        # infer if to load
        if len(self.load) > 0:
            self.load = True
            self.load_path = self.args.load
        else:
            self.load = False
        
        # default expr dir
        self.expr_dir = "expr/" + self.task + ("_%.4f_" % self.lr) + str(self.n_iter)

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

class SConfig(BaseConfig):
    def __init__(self):
        super(SConfig, self).__init__()
    
        self.parser.add_argument("--dataset", default="datasets/stylegan", help="The path to segmentation dataset")
        self.parser.add_argument("--seg", default=1., type=float, help="Coefficient of segmentation loss")
        self.parser.add_argument("--mse", default=1., type=float, help="Coefficient of MSE loss")
        self.parser.add_argument("--seg-cfg", default="conv1-19", help="Configure of segmantic segmentation extractor")
        self.parser.add_argument("--imsize", default=512, help="Train image and label size")
    
    def parse(self):
        super(SConfig, self).parse()
        self.dataset_path = self.args.dataset
        self.mse_coef = self.args.mse
        self.seg_coef = self.args.seg
        self.semantic_config = self.args.seg_cfg
        self.imsize = self.args.imsize
        if self.task == "seg": 
            self.ds = dataset.LatentSegmentationDataset(self.dataset_path, size=self.imsize)
            self.dl = DataLoader(self.ds,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=True)
        self.record = {'loss': [], 'mseloss': [], 'segloss': []}

        self.name = self.task + "_" + str(self.seg_coef) + "_" + self.semantic_config
        self.expr_dir = osj("expr", self.name)
        os.system("rm -r %s" % self.expr_dir)
        os.system("mkdir %s" % self.expr_dir)

    def print_info(self):
        super(SConfig, self).print_info()
        print("=> Segmentation configure: %s" % self.semantic_config)
        print("=> Segmentation loss coefficient: %.4f" % self.seg_coef)
        print("=> MSE loss coefficient: %.4f" % self.mse_coef)
        print("=> Segmentation dataset %s" % self.dataset_path)
        if self.task == "seg":
            print("=> Number samples: %d" % len(self.ds))
# python train/tssegtrain.py --task tsseg --gpu 3,5 --batch_size 4 --seg 0.2 --mse 10.0 --seg-cfg conv1-19
# python train/tssegtrain.py --task tsseg --gpu 2,6 --batch_size 4 --seg 0.2 --mse 10.0 --seg-cfg conv2-19
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
