import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


class MultiResolutionConvolution(nn.Module):
    def __init__(self, in_dims=[512, 512, 256, 64, 32, 16], out_dim=16, kernel_size=1):
        super(MultiResolutionConvolution, self).__init__()
        self.convs = nn.ModuleList()
        self.ksize = kernel_size
        self.in_dims = in_dims
        self.weight = nn.Parameter(torch.zeros(out_dim, sum(in_dims), 1, 1))
        torch.nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(len(in_dims), out_dim))
    
    def forward(self, x):
        """
        x is list of multi resolution feature map
        """
        outs = []
        prev = 0
        for i in range(len(self.in_dims)):
            cur = prev + self.in_dims[i]
            outs.append(F.conv2d(x[i], self.weight[:, prev : cur], self.bias[i]))
            prev = cur
        
        size = max([out.size(3) for out in outs])

        return sum([F.interpolate(out, size, mode="bilinear") for out in outs])


class ExtendedConv(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), bias=True, args=""):
        super().__init__()
        self.args = args
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) # He init
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * he_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight
        if "positive" in self.args:
            w = F.relu(w)
        x = F.conv2d(x, w, self.bias, padding=self.kernel_size // 2)
        return x


def get_bn(name, dim):
    if name == 'none':
        return None
    elif name == 'bn':
        return nn.BatchNorm2d(dim)
    elif name == 'in':
        return nn.InstanceNorm2d(dim)

class Generator(nn.Module):
    def __init__(self, out_dim=3, out_act="tanh", size=64, **kwargs):
        """
        Start from 4x4, upsample=3 -> 32
        """
        super(Generator, self).__init__()
        self.upsample = int(np.log2(size)) - 2
        dims = [64 * (2**i) for i in range(self.upsample+1)][::-1]
        self.out_dim = out_dim
        self.out_act = out_act
        self.dims = dims
        self.ksize = 1
        self.padsize = (self.ksize - 1) // 2

        self.fc = nn.Linear(128, 4 * 4 * dims[0])
        self.relu = nn.ReLU(True)
        self.deconvs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.upsample = nn.UpsamplingNearest2d(scale_factor=2)
            conv.conv = nn.Conv2d(prevDim, curDim, 3, 1, padding=1)
            conv.relu = nn.ReLU(True)
            self.deconvs.append(conv)
        self.visualize = nn.Conv2d(dims[-1], self.out_dim, 3, padding=1)
        self.tanh = nn.Tanh()

    def get_stage(self, x, detach=False):
        x = self.relu(self.fc(x)).view(-1, self.dims[0], 4, 4)

        stage = []
        for layers in self.deconvs:
            x = layers(x)
            if detach:
                stage.append(x.detach())
            else:
                stage.append(x)

        x = self.visualize(x)
        if self.out_act == "tanh":
            x = self.tanh(x)

        return x, stage

    def forward(self, x, seg=True, detach=False):
        x = self.relu(self.fc(x)).view(-1, self.dims[0], 4, 4)

        stage = []
        for layers in self.deconvs:
            x = layers(x)
            if detach:
                stage.append(x.detach())
            else:
                stage.append(x)
        self.stage = stage
        x = self.visualize(x)
        if self.out_act == "tanh":
            x = self.tanh(x)
            
        return x


class Discriminator(nn.Module):
    def __init__(self, size=64, in_dim=3, **kwargs):
        super(Discriminator, self).__init__()
        self.upsample = int(np.log2(size)) - 2
        dims = [64 * (2**i) for i in range(self.upsample+1)]
        self.dims = dims

        self.conv = nn.Conv2d(in_dim, dims[0], 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.convs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.conv = nn.Conv2d(prevDim, curDim, 4, stride=2, padding=1)
            conv.lrelu = nn.LeakyReLU(0.2, True)
            self.convs.append(conv)
        self.fc = nn.Linear(4 * 4 * dims[-1], 1)
        #self.sigmoid = nn.Sigmoid()

    def set_semantic_config(self, cfg):
        self.segcfg, self.n_class = cfg.split("-")
        self.n_class = int(self.n_class)

        if self.segcfg == "lowcat":
            org_conv = self.conv
            in_dim, out_dim = org_conv.in_channels, org_conv.out_channels
            ks = org_conv.kernel_size[0]
            new_conv = nn.Conv2d(self.n_class + in_dim, out_dim, ks)
            # need to do this, otherwise weight is not present, only weight_orig
            new_conv(torch.rand(1, self.n_class + in_dim, ks, ks))
            org_conv(torch.rand(1, in_dim, ks, ks))
            new_conv.weight.data[:, :in_dim] = org_conv.weight.data.detach().clone()
            new_conv.bias.data = org_conv.bias.data.detach().clone()
            self.conv = new_conv

    def forward(self, x):
        x = self.lrelu(self.conv(x))
        for layers in self.convs:
            x = layers(x)
        x = self.fc(x.view(-1, 4 * 4 * self.dims[-1]))
        return x