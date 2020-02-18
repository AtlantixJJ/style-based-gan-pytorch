import torch
import torch.nn.functional as F


class LinearSemanticExtractor(torch.nn.Module):
    """
    Extract the semantic segmentation from internal representation using 1x1 conv.
    Args:
        n_class:    The predict class number
        dims:       The dimenstions of a list of feature maps
    """
    def __init__(self, n_class, dims=[]):
        self.n_class = n_class
        self.dims = dims
        self.build_extractor_conv()
    
    def build_extractor_conv(self):
        def conv_block(in_dim, out_dim, ksize):
            _m = [torch.nn.Conv2d(in_dim, out_dim, ksize, bias=False)]
            return torch.nn.Sequential(*_m)

        self.semantic_extractor = torch.nn.ModuleList([
            conv_block(dim, self.n_class, 1)
                for dim in self.dims]).to(self.device)
        
        self.optim = torch.optim.Adam(self.semantic_extractor.parameters(), lr=1e-3)

    # The output is series of segmentation maps: individual layers, aggregated sum of layers
    # The last one is the final formal segmentation
    def forward(self, stage):
        count = 0
        outputs = []
        for i, seg_input in enumerate(stage):
            outputs.append(self.semantic_extractor[count](seg_input))
            count += 1
        size = outputs[-1].shape[2]

        # summation series
        for i in range(1, len(stage)):
            size = stage[i].shape[2]
            layers = [F.interpolate(s, size=size, mode="bilinear")
                for s in outputs[:i]]
            sum_layers = sum(layers) + outputs[i]
            outputs.append(sum_layers)

        return outputs