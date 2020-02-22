import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_semantic_extractor(config):
    if config == "linear":
        return LinearSemanticExtractor
    elif config == "nonlinear":
        return NonLinearSemanticExtractor
    elif config == "generative":
        return GenerativeSemanticExtractor
    elif config == "cascade":
        return CascadeSemanticExtractor



class BaseSemanticExtractor(nn.Module):
    def __init__(self, n_class, dims=[], mapid=None, category_groups=None):
        super().__init__()
        self.mapid = mapid
        self.n_class = n_class
        self.dims = dims
        if category_groups is None:
            category_groups = [[0, self.n_class]]
        self.category_groups = category_groups
        self.segments = [0] + list(np.cumsum(self.dims))
        
    def predict(self, stage):
        res = self.forward(stage, True)[0].argmax(1)
        if self.mapid:
            res = self.mapid(res)
        return res.detach().cpu().numpy().astype("int32")


class GenerativeSemanticExtractor(BaseSemanticExtractor):
    """
    Generative model like semantic extractor. Doesn't support multilabel segmentation.
    Args:
        n_layer:    number of convolutional layers to transform
                    generative representation to semantic embedding
        ksize:      kernel size of convolution
    """
    def __init__(self, ksize=3, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.ksize = ksize
        self.build()

    def build(self):
        def conv_block(in_dim, out_dim):
            midim = (in_dim + out_dim) // 2
            if self.n_layer == 1:
                _m = [nn.Conv2d(in_dim, out_dim, self.ksize), nn.ReLU(inplace=True)]
            else:
                _m = []
                _m.append(nn.Conv2d(in_dim, midim, self.ksize))
                _m.append(nn.ReLU(inplace=True))
                for i in range(self.n_layer - 2):
                    _m.append(nn.Conv2d(midim, midim, self.ksize))
                    _m.append(nn.ReLU(inplace=True))
                _m.append(nn.Conv2d(midim, out_dim, self.ksize))
                _m.append(nn.ReLU(inplace=True))
            return nn.Sequential(*_m)

        # transform generative representation to semantic embedding
        semantic_extractor = nn.ModuleList([
            conv_block(dim, dim) for dim in self.dims])
        # learning residual between different layers of generative representation
        semantic_reviser = nn.ModuleList([
            conv_block(prev, cur)
            for prev, cur in zip(self.dims[:-1], self.dims[1:])])
        # transform semantic embedding to label
        semantic_visualizer = nn.ModuleList([
            nn.Conv2d(dim, self.n_class, self.ksize) for dim in self.dims])

        self.semantic_branch = nn.ModuleList([
            semantic_extractor,
            semantic_reviser,
            semantic_visualizer])
        self.optim = torch.optim.Adam(self.semantic_branch.parameters(), lr=1e-3)

    def forward(self, stage, last_only=True):
        extractor, reviser, visualizer = self.semantic_branch
        hidden = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            if i == 0:
                hidden = extractor[i](seg_input)
            else:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = reviser[i - 1](hidden)
                hidden = hidden + extractor[i](seg_input)
            if not last_only:
                outputs.append(visualizer[i](hidden))
        if last_only:
            return [visualizer[-1](hidden)]
        return outputs


class LinearSemanticExtractor(BaseSemanticExtractor):
    """
    Extract the semantic segmentation from internal representation using 1x1 conv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def build(self):
        def conv_block(in_dim, out_dim, ksize):
            _m = [nn.Conv2d(in_dim, out_dim, ksize, bias=False)]
            return nn.Sequential(*_m)

        self.semantic_extractor = nn.ModuleList([
            conv_block(dim, self.n_class, 1)
                for dim in self.dims])
        
        self.optim = torch.optim.Adam(self.semantic_extractor.parameters(), lr=1e-3)

    def forward_category_group(self, stage, last_only=False):
        outputs = [[] for _ in range(len(self.category_groups))]

        for i, seg_input in enumerate(stage):
            x = self.semantic_extractor[i](seg_input)
            for cat_id, (bg, ed) in enumerate(self.category_groups):
                outputs[cat_id].append(x[:, bg:ed])

        size = outputs[0][-1].shape[2]

        if last_only:
            res = []
            for cat_id in range(len(self.category_groups)):
                r = [F.interpolate(s, size=size, mode="bilinear")
                    for s in outputs[cat_id]]
                res.append(sum(r))
            return res

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            for cat_id in range(len(self.category_groups)):
                layers = [F.interpolate(s, size=size, mode="bilinear")
                    for s in outputs[cat_id][:i]]
                sum_layers = sum(layers) + outputs[cat_id][i]
                outputs[cat_id].append(sum_layers)

        return outputs
        
    def forward_single_group(self, stage, last_only=False):
        outputs = []
        for i, seg_input in enumerate(stage):
            outputs.append(self.semantic_extractor[i](seg_input))
        size = outputs[-1].shape[2]

        if last_only:
            size = stage[-1].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear") for s in outputs]
            return [sum(layers)]

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers) + outputs[i]
            outputs.append(sum_layers)

        return outputs

    def forward(self, stage, last_only=False):
        if len(self.category_groups) == 1:
            return self.forward_single_group(stage, last_only)
        else:
            return self.forward_category_group(stage, last_only)

    """
    def copy_weight_from(self, coef):
        coef = torch.from_numpy(coef).view(coef.shape[0], coef.shape[1], 1, 1)
        for i, conv in enumerate(self.semantic_extractor.children()):
            prev_dim, cur_dim = self.segments[i], self.segments[i+1]
            state_dict = conv[0].state_dict()
            device = state_dict["weight"].device
            state_dict["weight"] = nn.Parameter(coef[:, prev_dim:cur_dim]).to(device)
            conv[0].load_state_dict(state_dict)
    """


class NonLinearSemanticExtractor(LinearSemanticExtractor):
    """
    Extract the semantic segmentation from internal representation using 1x1 conv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def build(self):
        def conv_block(in_dim, out_dim, ksize):
            midim = (in_dim + out_dim) // 2
            _m = []
            _m.append(nn.Conv2d(in_dim, midim, ksize))
            _m.append(nn.ReLU(inplace=True))
            _m.append(nn.Conv2d(midim, midim, ksize))
            _m.append(nn.Conv2d(midim, out_dim, ksize))
            return nn.Sequential(*_m)

        self.semantic_extractor = nn.ModuleList([
            conv_block(dim, self.n_class, 1)
                for dim in self.dims])
        
        self.optim = torch.optim.Adam(self.semantic_extractor.parameters(), lr=1e-3)


"""
class OVOLinearSemanticExtractor(nn.Module):
    def __init__(self, n_class, dims=[], mapid=None):
        super(OVOLinearSemanticExtractor, self).__init__()
        self.n_class = n_class
        self.mapid = mapid
        self.n_ovo = int(self.n_class * (self.n_class - 1) / 2)
        self.dims = dims
        self.segments = [0] + list(np.cumsum(self.dims))
        self.build_extractor_conv()

    def build_extractor_conv(self):
        def conv_block(in_dim, out_dim, ksize):
            _m = [nn.Conv2d(in_dim, out_dim, ksize)]
            return nn.Sequential(*_m)

        self.semantic_extractor = nn.ModuleList([
            conv_block(dim, self.n_ovo, 1)
                for dim in self.dims])
        
        self.optim = torch.optim.Adam(self.semantic_extractor.parameters(), lr=1e-3)

    def copy_weight_from(self, coef, intercept):
        coef = torch.from_numpy(coef).view(coef.shape[0], coef.shape[1], 1, 1)
        intercept = torch.from_numpy(intercept)
        zeros = torch.zeros_like(intercept)
        for i, conv in enumerate(self.semantic_extractor.children()):
            prev_dim, cur_dim = self.segments[i], self.segments[i+1]

            state_dict = conv[0].state_dict()
            device = state_dict["weight"].device
            state_dict["weight"] = nn.Parameter(coef[:, prev_dim:cur_dim]).to(device)
            if i == len(self.dims) - 1:
                state_dict["bias"] = nn.Parameter(intercept).to(device)
            else:
                state_dict["bias"] = nn.Parameter(zeros).to(device)
            conv[0].load_state_dict(state_dict)

    def predict(self, stage, last_only=False):
        res = self.forward(stage, True)[0].argmax(1)
        if self.mapid:
            res = self.mapid(res)
        return res.detach().cpu().numpy().astype("int32")

    # The output is series of segmentation maps: individual layers, aggregated sum of layers
    # The last one is the final formal segmentation
    def forward(self, stage, last_only=False):
        count = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            outputs.append(self.semantic_extractor[count](seg_input))
            count += 1
        size = outputs[-1].shape[2]

        if last_only:
            size = stage[-1].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear") for s in outputs]
            outputs = [sum(layers)]
        else:
            # summation series
            for i in range(1, len(stage)):
                size = stage[i].shape[2]
                layers = [F.interpolate(s, size=size, mode="bilinear")
                    for s in outputs[:i]]
                sum_layers = sum(layers) + outputs[i]
                outputs.append(sum_layers)

        # take votes, note that this will not be differentiable
        for k in range(len(outputs)):
            o = outputs[k] # summation of (N, n*(n-1)/2, H, W)
            N, No, H, W = o.shape
            res = torch.zeros(N, self.n_class, H, W)
            count = 0
            for i in range(self.n_class):
                for j in range(i+1, self.n_class):
                    mask = (o[:, count, :, :] > 0).bool()
                    res[:, i, :, :][mask] += 1
                    res[:, j, :, :][~mask] += 1
                    count += 1
            outputs[k] = res

        return outputs
"""