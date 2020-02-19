import torch
import torch.nn.functional as F
import numpy as np


class OVOLinearSemanticExtractor(torch.nn.Module):
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
            _m = [torch.nn.Conv2d(in_dim, out_dim, ksize)]
            return torch.nn.Sequential(*_m)

        self.semantic_extractor = torch.nn.ModuleList([
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
            state_dict["weight"] = torch.nn.Parameter(coef[:, prev_dim:cur_dim]).to(device)
            if i == len(self.dims) - 1:
                state_dict["bias"] = torch.nn.Parameter(intercept).to(device)
            else:
                state_dict["bias"] = torch.nn.Parameter(zeros).to(device)
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


class LinearSemanticExtractor(torch.nn.Module):
    """
    Extract the semantic segmentation from internal representation using 1x1 conv.
    Args:
        n_class:    The predict class number
        dims:       The dimenstions of a list of feature maps
    """
    def __init__(self, n_class, dims=[], mapid=None):
        super(LinearSemanticExtractor, self).__init__()
        self.mapid = mapid
        self.n_class = n_class
        self.dims = dims
        self.segments = [0] + list(np.cumsum(self.dims))
        self.build_extractor_conv()

    def copy_weight_from(self, coef):
        coef = torch.from_numpy(coef).view(coef.shape[0], coef.shape[1], 1, 1)
        for i, conv in enumerate(self.semantic_extractor.children()):
            prev_dim, cur_dim = self.segments[i], self.segments[i+1]
            state_dict = conv[0].state_dict()
            device = state_dict["weight"].device
            state_dict["weight"] = torch.nn.Parameter(coef[:, prev_dim:cur_dim]).to(device)
            conv[0].load_state_dict(state_dict)

    def build_extractor_conv(self):
        def conv_block(in_dim, out_dim, ksize):
            _m = [torch.nn.Conv2d(in_dim, out_dim, ksize, bias=False)]
            return torch.nn.Sequential(*_m)

        self.semantic_extractor = torch.nn.ModuleList([
            conv_block(dim, self.n_class, 1)
                for dim in self.dims])
        
        self.optim = torch.optim.Adam(self.semantic_extractor.parameters(), lr=1e-3)

    def predict(self, stage):
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
            return [sum(layers)]

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers) + outputs[i]
            outputs.append(sum_layers)

        return outputs
