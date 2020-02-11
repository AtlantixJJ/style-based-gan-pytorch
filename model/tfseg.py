import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import numpy as np


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5) # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True,
                intermediate=None, upscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        
        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1,1,1,1))
            w = w[:, :, 1:, 1:]+ w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)
    
        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size//2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size//2)
        
        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class MultiResolutionMLP(nn.Module):
    def __init__(self,
        in_dims=[512, 512, 256, 64, 32, 16],
        hidden_dims=[512],
        out_dim=16,
        kernel_size=1,
        args="l2_bias"):
        super(MultiResolutionMLP, self).__init__()
        self.args = args
        self.ksize = kernel_size
        self.in_dims = in_dims
        self.segments = []
        cur = 0
        for dim in in_dims:
            self.segments.append((cur, cur + dim))
            cur += dim

        self.in_weight = nn.Parameter(torch.zeros(out_dim, sum(in_dims), 1, 1))
        torch.nn.init.kaiming_normal_(self.weight)
        self.in_bias = nn.Parameter(torch.zeros(len(in_dims), hidden_dims[0]))

        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()

        prev_dim = sum(in_dims)
        for cur_dim in hidden_dims:
            weight = torch.zeros(cur_dim, prev_dim, 1, 1)
            torch.nn.init.kaiming_normal_(weight)
            self.hidden_weights.append(nn.Parameter(weight))
            bias = torch.zeros(prev_dim, cur_dim)
            self.hidden_biases.append(nn.Parameter(bias))
            prev_dim = cur_dim

    # (256, 1024, 1024) 16, 32, 64, 128 (128)
    def forward(self, x):
        """
        x is list of multi resolution feature map
        """
        outs = []
        prev = 0

        for i in range(len(self.in_dims)):
            cur = prev + self.in_dims[i]
            y = 0
            if "bias" in self.args:
                y = F.conv2d(x[i], self.weight[:, prev : cur], self.bias[i])
            else:
                y = F.conv2d(x[i], self.weight[:, prev : cur])
            outs.append(y)

            prev = cur
        
        size = max([out.size(3) for out in outs])

        mode = 0
        if "bilinear" in self.args:
            mode = "bilinear"
        elif "nearest" in self.args:
            mode = "nearest"

        return sum([F.interpolate(out, size, mode=mode) for out in outs])


class MultiResolutionConvolution(nn.Module):
    def __init__(self, in_dims=[512, 512, 256, 64, 32, 16], out_dim=16,
        kernel_size=1, args="l2_bias"):
        super(MultiResolutionConvolution, self).__init__()
        self.args = args
        self.mode = "nearest" if "nearest" in self.args else "bilinear"
        self.ksize = kernel_size
        self.in_dims = in_dims
        self.segments = []
        cur = 0
        for dim in in_dims:
            self.segments.append((cur, cur + dim))
            cur += dim
        self.weight = nn.Parameter(torch.zeros(out_dim, sum(in_dims), 1, 1))
        torch.nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(len(in_dims), out_dim))
    
    def get_orthogonal_weight(self):
        W = self.weight[:, :, 0, 0].permute(1, 0) # (1520, 16)
        self.Ms = []
        for i in range(len(self.segments)):
            bg, ed = self.segments[i]
            trunc_Q, trunc_R = torch.qr(W[bg:ed])
            I = torch.eye(ed - bg, ed - bg).cuda()
            M = I - torch.matmul(trunc_Q, trunc_Q.permute(1, 0))
            M = M.permute(1, 0).unsqueeze(2).unsqueeze(3) # (dim, dim, 1, 1)
            self.Ms.append(M)

    # orthogonalize the delta
    def orthogonalize(self, i, delta):
        new_delta = F.conv2d(delta, self.Ms[i])
        print("%.3f %.3f" % (new_delta.min(), new_delta.max()))
        return new_delta

    def forward(self, x):
        """
        x is list of multi resolution feature map
        """
        outs = []
        prev = 0

        for i in range(len(self.in_dims)):
            cur = prev + self.in_dims[i]
            y = 0
            if "bias" in self.args:
                y = F.conv2d(x[i], self.weight[:, prev : cur], self.bias[i])
            else:
                y = F.conv2d(x[i], self.weight[:, prev : cur])
            outs.append(y)

            prev = cur
        
        size = max([out.size(3) for out in outs])

        return sum([F.interpolate(out, size, mode=self.mode) for out in outs])

class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None
    
    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)
        
    def forward(self, x, latent):
        style = self.lin(latent) # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel=[1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride
    
    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2)-1)/2),
            groups=x.size(1)
        )
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor
    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))
        
    def simple_forward(self, x):
        return super().forward(x)

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x

class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer('avg_latent', avg_latent)
    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""
    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNorm()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None
    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(dlatent_size, nf*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        
    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer)
            
    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class G_synthesis(nn.Module):
    def __init__(self,
        dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
        num_channels        = 3,            # Number of output color channels.
        resolution          = 1024,         # Output resolution.
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        use_styles          = True,         # Enable style inputs?
        const_input_layer   = True,         # First layer is a learned constant?
        use_noise           = True,         # Enable noise inputs?
        randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
        use_wscale          = True,         # Enable equalized learning rate?
        use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
        use_instance_norm   = True,         # Enable instance normalization?
        dtype               = torch.float32,  # Data type to use for activations and outputs.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        ):
        
        super().__init__()
        self.ortho_w = None
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        #self.torgbs = nn.ModuleList()
        blocks = []
        self.stage = []
        for res in range(2, resolution_log2 + 1):
            self.stage.append(0)
            channels = nf(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                      use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
                
            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            last_channels = channels
            #if res != resolution_log2:
            #    self.torgbs.append(MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        #self.torgbs.append(self.torgb)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        
    def forward(self, dlatents_in, orthogonal=None, ortho_bias=0):
        cnt = 0
        stage = []    
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2])
            else:
                x = m(x, dlatents_in[:, 2*i:2*i+2])
            if orthogonal is not None and x.shape[3] >= 16 and x.shape[3] <= 256:
                x = ortho_bias[i] + orthogonal(cnt, x - ortho_bias[i])
                cnt += 1
            stage.append(x)
        rgb = self.torgb(x)
        return rgb, stage


class StyledGenerator(nn.Module):
    def __init__(self, semantic="mul-16"):
        super().__init__()
        self.g_mapping = G_mapping()
        self.g_synthesis = G_synthesis()

        self.segcfg, self.n_class, self.args = semantic.split("-")
        self.n_class = int(self.n_class)

        self.resample_mode = "nearest" if "nearest" in self.args else "bilinear"

        self.ksize = 1
        self.n_layer = 1
        self.semantic_dim = 64
        self.padsize = (self.ksize - 1) // 2

        if "conv" in self.segcfg:
            self.n_layer = int(self.args.split("_")[0])
            # final layer not good, but good classification on early layers
            self.build_conv_extractor()
        elif "cas" in self.segcfg:
            # good in general, but some details are still lost
            self.build_cascade_extractor()
        elif "gen" in self.segcfg:
            self.ksize = 3
            # generally the same with cas, a little worse, some details are lost
            self.build_gen_extractor()
        elif "cat" in self.segcfg:
            self.build_cat_extractor()
        elif "mul" in self.segcfg:
            ind = self.args.find("sl") # staring layer
            self.full_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
            self.mul_start_layer = int(self.args[ind + 2])
            self.build_multi_extractor()

    def build_multi_extractor(self):
        self.semantic_branch = MultiResolutionConvolution(
            #        4      8   16  32   64   128 256 512 1024
            in_dims=self.full_dims[self.mul_start_layer:],
            out_dim=self.n_class,
            kernel_size=self.ksize,
            args=self.args
        )

    def build_gen_extractor(self):
        def conv_block(in_dim, out_dim):
            midim = (in_dim + out_dim) // 2
            if self.n_layer == 1:
                _m = [MyConv2d(in_dim, out_dim, self.ksize), nn.ReLU(inplace=True)]
            else:
                _m = []
                _m.append(MyConv2d(in_dim, midim, self.ksize))
                _m.append(nn.ReLU(inplace=True))
                for i in range(self.n_layer - 2):
                    _m.append(MyConv2d(midim, midim, self.ksize))
                    _m.append(nn.ReLU(inplace=True))
                _m.append(MyConv2d(midim, out_dim, self.ksize))
                _m.append(nn.ReLU(inplace=True))
            return nn.Sequential(*_m)

        semantic_extractor = nn.ModuleList([
            conv_block(512, 512), # 4
            conv_block(512, 512), # 8
            conv_block(512, 512), # 16
            conv_block(512, 512), # 32
            conv_block(256, 256), # 64
            conv_block(128, 128), # 128
            conv_block(64 , 64 ), # 256
            conv_block(32 , 32 ), # 512
            conv_block(16 , 16 ) # 1024
        ])
        # start from 64
        semantic_visualizer = nn.ModuleList([
            MyConv2d(256, self.n_class, 1),
            MyConv2d(128, self.n_class, 1),
            MyConv2d(64 , self.n_class, 1),
            MyConv2d(32 , self.n_class, 1),
            MyConv2d(16 , self.n_class, 1)])
        
        semantic_reviser = nn.ModuleList([
            conv_block(512, 512),
            conv_block(512, 512),
            conv_block(512, 512),
            conv_block(512, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64 ),
            conv_block(64 , 32 ),
            conv_block(32 , 16 ),
            conv_block(16 , 16 )])

        self.semantic_branch = nn.ModuleList([
            semantic_extractor, # 0
            semantic_reviser, # 1
            semantic_visualizer]) # 2

    def build_cascade_extractor(self):
        def conv_block(in_dim, out_dim, ksize):
            midim = (in_dim + out_dim) // 2
            if self.n_layer == 1:
                _m = [MyConv2d(in_dim, out_dim, ksize), nn.ReLU(inplace=True)]
            else:
                _m = []
                _m.append(MyConv2d(in_dim, midim, ksize))
                _m.append(nn.ReLU(inplace=True))
                for i in range(self.n_layer - 2):
                    _m.append(MyConv2d(midim, midim, ksize))
                    _m.append(nn.ReLU(inplace=True))
                _m.append(MyConv2d(midim, out_dim, ksize))
                _m.append(nn.ReLU(inplace=True))
            return nn.Sequential(*_m)

        semantic_extractor = nn.ModuleList([
            conv_block(512, self.semantic_dim, self.ksize),
            conv_block(512, self.semantic_dim, self.ksize),
            conv_block(512, self.semantic_dim, self.ksize),
            conv_block(512, self.semantic_dim, self.ksize),
            conv_block(256, self.semantic_dim, self.ksize),
            conv_block(128, self.semantic_dim, self.ksize),
            conv_block(64 , self.semantic_dim, self.ksize),
            conv_block(32 , self.semantic_dim, self.ksize) # 512
        ])
        semantic_reviser = nn.ModuleList([
            conv_block(self.semantic_dim, self.semantic_dim, 3)
                for i in range(len(semantic_extractor) - 1)
            ])
        semantic_visualizer = nn.ModuleList([
            MyConv2d(self.semantic_dim, self.n_class, 1)
                for i in range(len(semantic_extractor) - 4)
            ])
        
        self.semantic_branch = nn.ModuleList([
            semantic_extractor, # 0
            semantic_reviser, # 1
            semantic_visualizer]) # 2

    def build_conv_extractor(self):
        def conv_block(in_dim, out_dim, ksize):
            if self.n_layer == 1:
                _m = [MyConv2d(in_dim, out_dim, ksize)]
            else:
                midim = (in_dim + out_dim) // 2
                _m = []
                _m.append(MyConv2d(in_dim, midim, ksize))
                for i in range(self.n_layer - 2):
                    _m.append(nn.ReLU(inplace=True))
                    _m.append(MyConv2d(midim, midim, ksize))
                _m.append(MyConv2d(midim, out_dim, ksize))
            return nn.Sequential(*_m)

        # start from 16x16 resolution
        self.semantic_extractor = nn.ModuleList([
            conv_block(512, self.n_class, self.ksize),
            conv_block(512, self.n_class, self.ksize),
            conv_block(512, self.n_class, self.ksize),
            conv_block(512, self.n_class, self.ksize),
            conv_block(256, self.n_class, self.ksize),
            conv_block(128, self.n_class, self.ksize),
            conv_block(64 , self.n_class, self.ksize),
            conv_block(32 , self.n_class, self.ksize),
            conv_block(16 , self.n_class, self.ksize)
        ])

        self.semantic_branch = self.semantic_extractor

    def extract_segmentation_gen(self, stage):
        extractor, reviser, visualizer = self.semantic_branch
        count = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            if i == 0:
                hidden = extractor[i](seg_input)
            else:
                hidden = F.interpolate(hidden, scale_factor=2, mode=self.resample_mode)
                hidden = reviser[i](hidden)
                hidden = hidden + extractor[i](seg_input)
            if seg_input.shape[3] >= 64:
                outputs.append(visualizer[count](hidden))
                count += 1
            
        return outputs

    def extract_segmentation_cascade(self, stage):
        extractor, reviser, visualizer = self.semantic_branch
        count = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            if seg_input.size(3) >= 1024:
                break
            
            if i == 0:
                hidden = extractor[i](seg_input)
            else:
                hidden = reviser[i - 1](F.interpolate(hidden, scale_factor=2))
                hidden = hidden + extractor[i](seg_input)
            if seg_input.size(3) >= 64:
                outputs.append(visualizer[count](hidden))
                count += 1
        return outputs
    
    def extract_segmentation_conv(self, stage):
        count = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            outputs.append(self.semantic_extractor[count](seg_input))
            count += 1
        return outputs

    def extract_segmentation_multi(self, stage):
        #for ind in range(len(stage)):
        #    if stage[ind].size(2) >= 16:
        #        break
        return [self.semantic_branch(stage[self.mul_start_layer:])]

    def freeze(self, train=False):
        self.freeze_g_mapping(train)
        self.freeze_g_synthesis(train)

    def freeze_g_mapping(self, train=False):
        for param in self.g_mapping.parameters():
            param.requires_grad = train

    def freeze_g_synthesis(self, train=False):
        for param in self.g_synthesis.parameters():
            param.requires_grad = train

    def set_noise(self, noises):
        if not hasattr(self, "noise_layers"):
            self.noise_layers = [l for n,l in self.named_modules() if "noise" in n]
        
        if noises is None:
            for i in range(len(self.noise_layers)):
                self.noise_layers[i].noise = None
            return

        for i in range(len(noises)):
            self.noise_layers[i].noise = noises[i]

    def forward(self, x, seg=True, detach=False, ortho_bias=None):
        if type(x) is list: # ML (1, 512)
            ws = [self.g_mapping.simple_forward(z) for z in x]
            x = torch.stack(ws, dim=1)
        elif x.shape[1] == 512: # LL
            x = self.g_mapping(x)
        elif x.shape[1] == 1: # GL
            x = x.expand(-1, 18, -1)
        elif x.shape[1] == 18: # EL
            x = x
        else:
            print("!> Error shape")

        if ortho_bias is not None:
            image, stage = self.g_synthesis(x,
                orthogonal=self.semantic_branch.orthogonalize,
                ortho_bias=ortho_bias)
        else:
            image, stage = self.g_synthesis(x)
        self.stage = stage

        if detach:
            for i in range(len(self.stage)):
                self.stage[i] = self.stage[i].detach()

        if seg:
            seg = self.extract_segmentation(stage)[-1]
            return image, seg
        return image
    
    def extract_segmentation(self, stage):
        if "conv" in self.segcfg:
            return self.extract_segmentation_conv(stage)
        elif "cas" in self.segcfg:
            return self.extract_segmentation_cascade(stage)
        elif "gen" in self.segcfg:
            return self.extract_segmentation_gen(stage)
        elif "cat" in self.segcfg:
            return self.extract_segmentation_cat(stage)
        elif "mul" in self.segcfg:
            return self.extract_segmentation_multi(stage)

    def predict(self, latent):
        # start from w+, output (512, 512)
        if latent.dim() == 3 and latent.shape[1] == 18:
            gen = self.g_synthesis(latent).clamp(-1, 1)
            #gen_np = gen[0].detach().cpu().numpy()
            #gen_np = ((gen_np + 1) * 127.5).transpose(1, 2, 0)
            seg_logit = self.extract_segmentation()[-1]
            seg_logit = F.interpolate(seg_logit, (512, 512), mode="bilinear")
            seg = seg_logit.argmax(dim=1)
            #seg = seg.numpy()
        return gen, seg