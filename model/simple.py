import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def get_bn(name, dim):
    if name == 'none':
        return None
    elif name == 'bn':
        return nn.BatchNorm2d(dim)
    elif name == 'in':
        return nn.InstanceNorm2d(dim)

class Generator(nn.Module):
    def __init__(self, bn='none', upsample=3, semantic=""):
        """
        Start from 4x4, upsample=3 -> 32
        """
        super(Generator, self).__init__()
        dims = [64 * (2**i) for i in range(upsample+1)][::-1]
        self.dims = dims
        self.upsample = upsample
        self.segcfg, self.semantic_dim, self.n_class = semantic.split("-")
        self.n_class = int(self.n_class)

        self.fc = nn.Linear(128, 4 * 4 * dims[0])
        self.relu = nn.ReLU(True)
        self.deconvs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.upsample = nn.UpsamplingNearest2d(scale_factor=2)
            conv.conv = nn.Conv2d(prevDim, curDim, 3, 1, padding=1)
            conv.bn = get_bn(bn, curDim)
            conv.relu = nn.ReLU(True)
            self.deconvs.append(conv)
        self.visualize = nn.Conv2d(dims[-1], 3, 3, padding=1)
        self.tanh = nn.Tanh()
        self.build_cat_extractor()

    def build_cat_extractor(self):
        def conv_block(in_dim, out_dim):
            _m = [nn.Conv2d(in_dim, out_dim, 1, 1), nn.ReLU(inplace=True)]
            return nn.Sequential(*_m)

        if self.upsample == 3:
            dims = [128, 64, 64] # 256
        elif self.upsample == 4:
            dims = [256, 128, 64, 64] # 512

        semantic_extractor = nn.ModuleList()
        for in_dim, out_dim in zip(self.dims[1:], dims):
            semantic_extractor.append(conv_block(in_dim, out_dim))

        semantic_visualizer = nn.Conv2d(sum(dims), self.n_class, 1, 1)

        self.semantic_branch = nn.ModuleList([
            semantic_extractor,
            semantic_visualizer])

    def extract_segmentation(self, stage):
        count = 0
        outputs = []
        hiddens = []
        extractor, visualizer = self.semantic_branch
        for i, seg_input in enumerate(stage):
            hiddens.append(extractor[count](seg_input))
            count += 1
        # concat
        base_size = hiddens[-1].size(2)
        feat = torch.cat(
            [F.interpolate(h.float(), base_size, mode="bilinear") for h in hiddens],
            1)
        outputs.append(visualizer(feat))
        return outputs

    def forward(self, x, seg=True):
        x = self.relu(self.fc(x)).view(-1, self.dims[0], 4, 4)

        self.stage = []
        for layers in self.deconvs:
            x = layers(x)
            self.stage.append(x)
        x = self.tanh(self.visualize(x))

        if seg:
            seg = self.extract_segmentation(self.stage)[-1]
            return x, seg
        return x


class Discriminator(nn.Module):
    def __init__(self, bn='none', upsample=3):
        super(Discriminator, self).__init__()

        dims = [64 * (2**i) for i in range(upsample+1)]
        self.dims = dims

        self.conv = nn.Conv2d(3, dims[0], 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.convs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.conv = nn.Conv2d(prevDim, curDim, 4, stride=2, padding=1)
            conv.bn = get_bn(bn, curDim)
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
            new_conv.weight.data[:, :in_dim] = org_conv.weight.data.clone()
            new_conv.bias.data = org_conv.bias.data.clone()
            self.conv = new_conv

    def forward(self, x):
        x = self.lrelu(self.conv(x))
        for layers in self.convs:
            x = layers(x)
        x = self.fc(x.view(-1, 4 * 4 * self.dims[-1]))
        return x