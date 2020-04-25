import torch
import torch.nn.functional as F


def l1(module):
    return sum([p.abs().sum() for p in module.parameters()])

def l1_pos(module):
    return sum([p[p > 0].sum() for p in module.parameters()])


def l1norm(module):
    res = 0
    for conv in module.children():
        w = conv[0].weight[:, :, 0, 0]
        res = res + w.abs().sum(1)
    return ((res - 1) ** 2).mean()


def kl_div(segs, prob):
    seglosses = []
    for s in segs:
        layer_loss = 0
        # label is large : downsample label
        if s.size(2) < prob.size(2): 
            l_ = F.interpolate(
                prob.unsqueeze(1), s.size(2),
                mode="bilinear",
                align_corners=True)
            layer_loss = F.kl_div(s, l_)
        # label is small : downsample seg
        elif s.size(2) >= prob.size(2): 
            s_ = F.interpolate(s, prob.size(2),
                mode="bilinear", align_corners=True)
            layer_loss = F.kl_div(s_, prob)
        seglosses.append(layer_loss)
    segloss = sum(seglosses[:-1]) * 0.1 + seglosses[-1]
    #print(segloss, segs.min(), segs.max(), prob.min(), prob.max())
    return segloss


def onehot(x, n):
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)

# multilabel non-exclusive classification
def bceloss(segs, ext_label):
    multihot = onehot(ext_label.unsqueeze(1).cpu(), segs[0].shape[1]).cuda()
    seglosses = []
    for s in segs:
        layer_loss = 0
        # label is large : downsample label
        if s.size(2) < multihot.size(2): 
            l_ = F.interpolate(multihot.unsqueeze(1), s.size(2),
                mode="bilinear", align_corners=True)
            layer_loss = F.binary_cross_entropy_with_logits(s, l_)
        # label is small : downsample seg
        elif s.size(2) >= multihot.size(2): 
            s_ = F.interpolate(s, multihot.size(2),
                mode="bilinear", align_corners=True)
            layer_loss = F.binary_cross_entropy_with_logits(s_, multihot)
        seglosses.append(layer_loss)
    segloss = sum(seglosses[:-1]) * 0.1 + seglosses[-1]
    #print(segloss, segs.min(), segs.max(), prob.min(), prob.max())
    return segloss


# segs : [(N, C, H, W)]
# ext_label : (N, H, W)
def segloss(segs, ext_label):
    seglosses = []
    for s in segs:
        layer_loss = 0
        # label is large : downsample label
        if s.size(2) < ext_label.size(2): 
            l_ = ext_label.float()
            l_ = F.interpolate(l_.unsqueeze(1), s.size(2),
                mode="nearest", align_corners=True).squeeze(1)
            layer_loss = F.cross_entropy(s, l_.long())
        # label is small : downsample seg
        elif s.size(2) > ext_label.size(2): 
            s_ = F.interpolate(s, ext_label.size(2),
                mode="bilinear", align_corners=True)
            layer_loss = F.cross_entropy(s_, ext_label)
        else:
            layer_loss = F.cross_entropy(s, ext_label)
        seglosses.append(layer_loss)
    segloss = sum(seglosses[:-1]) * 0.1 + seglosses[-1]
    return segloss