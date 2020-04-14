import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_semantic_extractor(config):
    if config == "linear":
        return LinearSemanticExtractor
    elif config == "spherical":
        return NormalizedLinearSemanticExtractor
    elif config == "unitnorm":
        return UnitNormalizedLinearSemanticExtractor
    elif config == "unit":
        return UnitLinearSemanticExtractor
    elif config == "projective":
        return ProjectiveLinearSemanticExtractor
    elif config == "nonlinear":
        return NonLinearSemanticExtractor
    elif config == "generative":
        return GenerativeSemanticExtractor


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
        self.ksize = ksize
        self.build()

    def build(self):
        def conv_block(in_dim, out_dim):
            midim = (in_dim + out_dim) // 2
            _m = [
                nn.Conv2d(in_dim, out_dim, self.ksize, 1, self.ksize // 2),
                nn.ReLU(inplace=True)]
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
            nn.Conv2d(dim, self.n_class, self.ksize, 1, self.ksize // 2) for dim in self.dims])

        self.semantic_branch = nn.ModuleList([
            semantic_extractor,
            semantic_reviser,
            semantic_visualizer])
        self.optim = torch.optim.Adam(self.semantic_branch.parameters(), lr=1e-3)

    def forward(self, stage, last_only=True):
        extractor, reviser, visualizer = self.semantic_branch
        hidden = 0
        outputs = []
        final_segmentation = 0
        for i, seg_input in enumerate(stage):
            if i == 0:
                hidden = extractor[i](seg_input)
            else:
                #print(hidden.shape, seg_input.shape)
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = reviser[i - 1](hidden)
                hidden = hidden + extractor[i](seg_input)
                
            if i + 1 == len(stage):
                final_segmentation = visualizer[i](hidden)
            else:
                outputs.append(visualizer[i](hidden))

        if last_only:
            return [final_segmentation]

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers)
            if i + 1 == len(stage):
                sum_layers = sum_layers + final_segmentation 
            else:
                sum_layers = sum_layers + outputs[i]
            outputs.append(sum_layers)
        outputs.append(final_segmentation)
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

    def forward_category_group(self, stage, last_only=True):
        outputs = [[] for _ in range(len(self.category_groups))]

        for i, seg_input in enumerate(stage):
            x = self.semantic_extractor[i](seg_input)
            for cat_id, (bg, ed) in enumerate(self.category_groups):
                outputs[cat_id].append(x[:, bg:ed])

        size = max([o.shape[2] for o in outputs[0]])

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
        
    def forward_single_group(self, stage, last_only=True):
        outputs = []
        for i, seg_input in enumerate(stage):
            outputs.append(self.semantic_extractor[i](seg_input))
        size = max([o.shape[2] for o in outputs])

        if last_only:
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs]
            return [sum(layers)]

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers) + outputs[i]
            outputs.append(sum_layers)

        return outputs

    def forward(self, stage, last_only=True):
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


class ProjectiveLinearSemanticExtractor(LinearSemanticExtractor):
    """
    Project the feature to 2-dim plane
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def build(self):
        semantic_projector = nn.ModuleList([
            nn.Conv2d(dim, 2, 1, bias=False)
                for dim in self.dims])
        semantic_visualizer = nn.Conv2d(2, self.n_class, 1, bias=False)
        self.semantic_branch = nn.ModuleList([
                semantic_projector,
                semantic_visualizer
            ])
        self.optim = torch.optim.Adam(self.semantic_branch.parameters(), lr=1e-3)

    def forward(self, stage, last_only=False, projection=False):
        projects = []
        semantic_projector, semantic_visualizer = self.semantic_branch
        for i, seg_input in enumerate(stage):
            proj = semantic_projector[i](seg_input)
            projects.append(proj)
        size = stage[-1].shape[2]
        project = sum([F.interpolate(p, size=size, mode="bilinear")
            for p in projects])
        seg = semantic_visualizer(project)
        if last_only:
            if projection:
                return [seg], project
            return [seg]

        # summation series
        """
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers) + outputs[i]
            outputs.append(sum_layers)

        return outputs
        """


class NormalizedLinearSemanticExtractor(BaseSemanticExtractor):
    """
    Extract the semantic segmentation from internal representation using 1x1 conv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()
    
    def build(self):
        self.weight = nn.Parameter(torch.zeros(self.n_class, sum(self.dims), 1, 1))
        torch.nn.init.kaiming_normal_(self.weight)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
       
    def forward(self, stage, last_only=True):
        maxsize = stage[-1].shape[2] // 2
        with torch.no_grad():
            feat = torch.cat([F.interpolate(s, size=maxsize, mode="bilinear")
                for s in stage], 1)
            feat = F.normalize(feat, 2, 1)

        if last_only:
            return [F.conv2d(feat, self.weight)]

        prev = 0
        # side outputs
        sides = []
        for i in range(len(self.dims)):
            cur = prev + self.dims[i]
            sides.append(F.conv2d(feat[:, prev:cur], self.weight[:, prev : cur]))
            prev = cur

        # cumulative results
        outputs = []
        x = 0
        for i in range(len(stage)):
            x = x + sides[i]
            size = min(stage[i].shape[2], sides[i].shape[2])
            if x.shape[3] != size:
                outputs.append(F.interpolate(x, size=size, mode="bilinear"))
            else:
                outputs.append(x)
        
        return outputs


class UnitNormalizedLinearSemanticExtractor(NormalizedLinearSemanticExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()
        
    def forward(self, stage, last_only=True):
        maxsize = min(max([s.shape[2] for s in stage]), 256)
        w = F.normalize(self.weight[:, :, 0, 0], 2, 1)
        w = w.view(-1, w.shape[1], 1, 1)

        with torch.no_grad():
            feat = torch.cat([F.interpolate(s, size=maxsize, mode="bilinear") for s in stage], 1)
            feat = F.normalize(feat, 2, 1)

        if last_only:
            res = F.conv2d(feat, w)
            mini, maxi = res.min(), res.max()
            res = 2 * ((res - mini) / (maxi - mini) - 0.5)
            return [5 * res]


class UnitLinearSemanticExtractor(BaseSemanticExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()
    
    def build(self):
        self.weight = nn.Parameter(torch.zeros(self.n_class, sum(self.dims), 1, 1))
        torch.nn.init.kaiming_normal_(self.weight)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, stage, last_only=True):
        w = F.normalize(self.weight[:, :, 0, 0], 2, 1)
        w = w.view(-1, w.shape[1], 1, 1)
        outputs = []
        if last_only:
            for i in range(len(self.segments) - 1):
                prev = self.segments[i]
                cur = self.segments[i+1]
                outputs.append(F.conv2d(stage[i], w[:, prev:cur]))
            maxsize = max([o.shape[2] for o in outputs])
            layers = [F.interpolate(o, size=maxsize, mode="bilinear")
                for o in outputs]
            return [sum(layers)]


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
