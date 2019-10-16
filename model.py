import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(
            input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(
            grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer(
            'weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel,
                                    kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel,
                                kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        self.gamma, self.beta = style.chunk(2, 1)

        out = self.norm(input)
        out = self.gamma * out + self.beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class AttentionModule(nn.Module):
    def __init__(self, att, style_dim, out_channel, mtd="cos", norm="ch"):
        super(AttentionModule, self).__init__()
        self.att = att
        self.mtd = mtd[:-1]
        self.norm = norm
        self.midims = 256

        if "cos" in mtd:
            self.GAMMA = 5
        else:
            self.GAMMA = 1

        if "gen" in mtd:
            layers = int(mtd[-1])
            if layers == 1:
                self.gen = EqualLinear(style_dim, out_channel * self.att + self.att)
            else:
                modules = [EqualLinear(style_dim, self.midims), nn.ReLU(inplace=True)]
                for i in range(layers - 1):
                    modules.append(EqualLinear(self.midims, self.midims))
                    modules.append(nn.ReLU(inplace=True))
                modules.append(
                    EqualLinear(self.midims, out_channel * self.att + self.att))
                self.gen = nn.Sequential(*modules)
        if "conv" in mtd:
            layers = int(mtd[-1])
            if layers == 1:
                self.conv_feat = EqualConv2d(out_channel, att, 1, 1)
                self.fc_c = EqualLinear(style_dim, att)
                self.conv_fuse = None
            else:
                self.conv_feat = EqualConv2d(out_channel, self.midims, 1, 1)
                self.fc_c = EqualLinear(style_dim, self.midims)
                modules = []
                for i in range(layers - 1):
                    modules.append(nn.ReLU(inplace=True))
                    modules.append(EqualConv2d(self.midims, self.midims, 1, 1))
                modules.append(nn.ReLU(inplace=True))
                modules.append(EqualConv2d(self.midims, self.att, 1, 1))
                self.conv_fuse = nn.Sequential(*modules)



    def normalize(self, mask):
        """
        Return list of masks [N, 1, H, W]
        """
        if self.norm == "ch":
            if type(mask) is list:
                mask = torch.cat(mask, 1)
            masks = F.softmax(self.GAMMA * mask, 1)
            masks = [m.unsqueeze(1) for m in torch.unbind(masks, dim=1)]
        elif self.norm == "sp":
            if type(mask) is list:
                mask = torch.cat(mask, 1)
            n, c, h, w = mask.shape
            masks = F.softmax(mask.view(n, c, h * w), 2).view(n, c, h, w)
            masks = [m.unsqueeze(1) for m in torch.unbind(masks, dim=1)]
        elif self.norm == "elt":
            if type(mask) is list:
                masks = [torch.nn.functional.sigmoid(m) for m in mask]
            else:
                masks = torch.nn.functional.sigmoid(mask)
                masks = [m.unsqueeze(1) for m in torch.unbind(masks, dim=1)]
        elif self.norm == "none":
            pass
        return masks

    def forward(self, feat, style):
        self.mask = []
        n, c, h, w = feat.shape

        """deprecated
        if self.mtd == "cos":
            masks = []
            self.keys = [linear(style).unsqueeze(2).unsqueeze(3)
                         for linear in self.linears]
            nfeat = feat / torch.sqrt((feat ** 2).sum(1, keepdim=True))
            for i in range(self.att):
                key = self.keys[i] / \
                    torch.sqrt((self.keys[i] ** 2).sum(1, keepdim=True))
                ip = (nfeat * key).sum(1, keepdim=True)
                masks.append(ip)
        """
        if self.mtd == "gen":
            res = self.gen(style)
            #self.gen_bias = res[:, -self.att:]
            self.gen_weight = res[:, :-self.att].view(n, self.att, c, 1, 1)
            masks = torch.cat([F.conv2d(feat[i:i+1], self.gen_weight[i])
                               for i in range(n)], 0)
        elif self.mtd == "gencos":
            nfeat = feat / torch.sqrt((feat ** 2).sum(1, keepdim=True))
            res = self.gen(style)
            #self.gen_bias = res[:, -self.att:]
            weight = res[:, :-self.att].view(n, self.att, c, 1, 1)
            weight_flt = weight.view(n, self.att, -1)
            weight_normalizer = torch.sqrt(
                (weight_flt ** 2).sum(2)).view(n, self.att, 1, 1, 1)
            self.gen_weight = weight / weight_normalizer
            masks = torch.cat([F.conv2d(nfeat[i:i+1], self.gen_weight[i])  # , self.gen_bias[i])
                               for i in range(n)], 0)

        elif self.mtd == "conv":
            f1 = self.conv_feat(feat)
            n, c, h, w = f1.shape
            f2 = self.fc_c(style)
            f = f1 + f2.view(n, c, 1, 1)
            if self.conv_fuse is not None:
                f = self.conv_fuse(f)
            masks = f

        self.mask = self.normalize(masks)

        return self.mask


class AttentionAdainModule(nn.Module):
    def __init__(self, att, style_dim, out_channel, adain=None, mtd="uni-cos-ch"):
        super(AttentionAdainModule, self).__init__()
        self.mtdargs = mtd.split("-")
        self.atthead = AttentionModule(
            att, style_dim, out_channel, self.mtdargs[1], self.mtdargs[2])

        if "uni" in self.mtdargs[0]:
            self.attadain = adain
        elif "sep" in self.mtdargs[0]:
            self.attadain = nn.ModuleList(
                [AdaptiveInstanceNorm(out_channel, style_dim) for i in range(att)])

    def forward(self, input, style, masks=None):
        # [N, 1, H, W] * [N, C, H, W]
        if self.mtdargs[0] == "unibfr":
            if masks is None:
                self.mask = self.atthead(input, style)
            else:
                self.mask = masks
            out = sum(self.mask) * self.attadain(input, style)
        elif self.mtdargs[0] == "uniaft":
            x = self.attadain(input, style)
            if masks is None:
                self.mask = self.atthead(x, style)
            else:
                self.mask = masks
            out = sum(self.mask) * x
        elif self.mtdargs[0] == "sep":
            if masks is None:
                self.mask = self.atthead(input, style)
            else:
                self.mask = masks
            out = sum([m * a(input, style)
                       for m, a in zip(self.mask, self.attadain)])
        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
        att=0,
        att_mtd="uni-cos",
        lerp=0
    ):
        super().__init__()
        self.att = att

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(
            out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

        if self.att > 0:
            N = self.att
            self.lerp = lerp
            self.attention1 = AttentionAdainModule(
                att, style_dim, out_channel, self.adain1, att_mtd)
            self.attention2 = AttentionAdainModule(
                att, style_dim, out_channel, self.adain2, att_mtd)

    def forward(self, input, style, noise, masks=[None, None], feat_list=None):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)

        if feat_list is not None:
            feat_list.append(["BeforeAdain1", out])

        if self.att > 0:
            new_out = self.attention1(out, style, masks=masks[0])
            old_out = self.adain1(out, style)
            out = self.lerp * new_out + (1 - self.lerp) * old_out
        else:
            out = self.adain1(out, style)

        if feat_list is not None:
            feat_list.append(["AfterAdain1", out])

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)

        if feat_list is not None:
            feat_list.append(["BeforeAdain2", out])

        if self.att > 0:
            new_out = self.attention2(out, style, masks=masks[1])
            old_out = self.adain2(out, style)
            out = self.lerp * new_out + (1 - self.lerp) * old_out
        else:
            out = self.adain2(out, style)

        if feat_list is not None:
            feat_list.append(["AfterAdain2", out])

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, fused=True, **kwargs):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True,
                                att=0),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True,
                                att=0),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True,
                                att=0),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True,
                                **kwargs),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True,
                                **kwargs),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True,
                                fused=fused, **kwargs),  # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True,
                                fused=fused, **kwargs),  # 256
                StyledConvBlock(64, 32, 3, 1, upsample=True,
                                fused=fused, **kwargs),  # 512
                StyledConvBlock(32, 16, 3, 1, upsample=True,
                                fused=fused, **kwargs),  # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def forward(self, style, noise, masks=None, step=0, alpha=-1, mixing_range=(-1, -1), feat_list=None):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if masks is not None:
                mask = masks[i]
            else:
                mask = [None, None]

            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else: 
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out
                out = conv(out, style_step, noise[i], mask, feat_list)
            else:
                out = conv(out, style_step, noise[i], mask, feat_list)

            if feat_list is not None:
                for k in range(1, 5):
                    feat_list[-k][0] = ("Progression%d/" % i) + feat_list[-k][0]

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(
                        skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8, **kwargs):
        super().__init__()

        self.generator = Generator(code_dim, **kwargs)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("=> Skip %s" % name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(
        self,
        input,
        noise=None,
        masks=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
        feat_list=None
    ):
        self.styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            self.styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size,
                                         size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in self.styles:
                styles_norm.append(
                    mean_style + style_weight * (style - mean_style))

            self.styles = styles_norm

        return self.generator(self.styles, noise, masks, step, alpha, mixing_range=mixing_range, feat_list=feat_list)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0 and i == step and 0 <= alpha < 1:
                skip_rgb = F.avg_pool2d(input, 2)
                skip_rgb = self.from_rgb[index + 1](skip_rgb)
                out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class WrappedStyledGenerator(nn.Module):
    """
    Standard wrapper for mask based spatial debug purpose.
    """

    def __init__(self, resume="./stylegan-512px-running-180000.model", gpu=-1, seed=None):
        super(WrappedStyledGenerator, self).__init__()
        print("=> Wrapped Style GAN initializing")
        self.code_dim = 512
        self.n_mlp = 8
        self.step = 7
        self.alpha = 1
        self.style_weight = 0.7
        self.device = 'cuda' if gpu >= 0 else 'cpu'
        self.resume = resume
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        print("=> Constructing network architecture")
        self.model = StyledGenerator(code_dim=self.code_dim, n_mlp=self.n_mlp)
        self.model = self.model.to(self.device)
        print("=> Loading parameter from %s" % self.resume)
        self.model.load_state_dict(torch.load(self.resume))
        self.model.eval()
        print("=> Check running")
        import os
        if not os.path.exists("mean_style.data"):
            self.mean_style = self.model.mean_style(
                torch.randn(1024, 512).to(self.device))
            torch.save(self.mean_style, "mean_style.data")
        else:
            self.mean_style = torch.load("mean_style.data")
            self.latent = torch.randn(1, self.code_dim).to(self.device)
            self.sample = self.model(self.latent,
                                     step=self.step,
                                     alpha=self.alpha,
                                     mean_style=self.mean_style,
                                     style_weight=self.style_weight)
        print("=> Initialization done")

    def forward(self, latent, ctrl_args=None):
        """
        Args:
            latent: numpy array of (1, 512)
            ctrl_args: [layer_ind, mask1, mask2]
        Returns:
            [n, h, w, ch] in numpy, [-1, 1] range
        """
        latent = torch.from_numpy(latent).float().to(self.device)
        if ctrl_args is None:
            self.sample = self.model(latent,
                                     step=self.step,
                                     alpha=self.alpha,
                                     mean_style=self.mean_style,
                                     style_weight=self.style_weight)
        else:
            mask = ctrl_args[1]
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            ctrl_args[1] = mask
            self.sample = self.model.dbg(latent,
                                         step=self.step,
                                         alpha=self.alpha,
                                         mean_style=self.mean_style,
                                         style_weight=self.style_weight,
                                         ctrl_args=ctrl_args)
        return self.sample.detach().cpu().numpy().transpose(0, 2, 3, 1)
