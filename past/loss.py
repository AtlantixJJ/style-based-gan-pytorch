import torch
import numpy as np
EPS = 1e-6


def maskarealoss(styledblocks, target=0.5, gamma=3, coef=1.0):
    """
    Masks: list of [N, 1, H, W]
    """
    def lossitem(x):
        # input is scalar
        if x < target:
            return torch.pow((target - x) * 1/target, gamma)
        else:
            return torch.pow((x - target) * 1/(1-target), gamma)

    masks = []
    loss = 0
    for blk in styledblocks:
        if blk.att > 0 and blk.attention1.mask is not None:
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    for m in masks:
        n, c, h, w = m.shape
        fm = m.view(n, -1).mean(1)
        for i in range(n):
            loss += lossitem(fm[i])
    loss = float(coef) * loss.mean() / len(m) / n
    return loss


def maskvalueloss(styledblocks, target=0.5, gamma=3, coef=1.0):
    """
    What about winner take all strategy?
    """
    def lossitem(x):
        noise = np.random.randn() * 1e-3
        return 1 - torch.pow(2 * (x - 0.5 + noise).abs(), gamma)
    loss = 0
    masks = []
    for blk in styledblocks:
        if blk.att > 0 and hasattr(blk.attention1, "mask"):
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    loss = coef * sum([lossitem(m).mean() for m in masks]) / len(masks)
    return loss


def maskdivloss(styledblocks, coef=1):
    def cross_similarity(masks):
        N = len(masks)
        n, c, h, w = masks[0].shape
        norm_masks = [torch.sqrt(
            (m.view(n, -1) ** 2).sum(1, keepdim=True) + EPS) for m in masks]

        nmasks = [m.view(n, -1) / nm
                  for m, nm in zip(masks, norm_masks)]
        cos = 0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                cos += (nmasks[i] * nmasks[j]).sum(1)  # [N,]
                count += 1
        return cos.mean() / count

    loss = 0
    count = 0
    for blk in styledblocks:
        if blk.att > 0 and hasattr(blk.attention1, "mask"):
            N = len(blk.attention1.mask)
            for masks in [blk.attention1.mask, blk.attention2.mask]:
                loss += cross_similarity(masks)
                count += 1

    loss = coef * loss / count
    return loss
