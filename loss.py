import torch
import torch.nn.functional as F

# segs : [(N, C, H, W)]
# ext_label : (N, H, W)
def segloss(segs, ext_label):
    seglosses = []
    for s in segs:
        layer_loss = 0
        # label is large : downsample label
        if s.size(2) < ext_label.size(2): 
            l_ = ext_label.float()
            l_ = F.interpolate(l_.unsqueeze(1), s.size(2), mode="nearest").squeeze(1)
            layer_loss = F.cross_entropy(s, l_.long())
        # label is small : downsample seg
        elif s.size(2) >= ext_label.size(2): 
            s_ = F.interpolate(s, ext_label.size(2), mode="bilinear")
            layer_loss = F.cross_entropy(s_, ext_label)
        seglosses.append(layer_loss)
    segloss = sum(seglosses[:-1]) * 0.1 + seglosses[-1]
    return segloss