import torch
import numpy as np
EPS = 1e-6


def select_max(x, y):
    if x < y:
        return y
    else:
        return x


def maskarealoss(styledblocks, target=0.5, gamma=3, coef=1.0):
    def lossitem(x):
        if x < target:
            return torch.pow((target - x) * 1/target, gamma)
        else:
            return torch.pow((x - target) * 1/(1-target), gamma)

    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    return float(coef) * sum([lossitem(m.mean()) for m in masks]) / len(masks)


def maskvalueloss(styledblocks, target=0.5, gamma=3, coef=1.0):
    """
    What about winner take all strategy?
    """
    def lossitem(x):
        noise = np.random.randn() * 1e-2
        return 1 - torch.pow(2 * (x - 0.5 + noise).abs(), gamma)
    loss = 0
    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    return coef * sum([lossitem(m).mean() for m in masks]) / len(masks)


def maskdivloss(styledblocks, theta=0.2, coef=1):
    loss = 0
    count = 0
    for blk in styledblocks:
        if blk.att > 0:
            N = len(blk.attention1.mask)
            for i in range(N):
                for j in range(i+1, N):
                    m1 = torch.sqrt(
                        blk.attention1.mask[i] * blk.attention1.mask[j])
                    m2 = torch.sqrt(
                        blk.attention2.mask[i] * blk.attention2.mask[j])
                    mask1 = (m1 > theta).type_as(m1)
                    mask2 = (m2 > theta).type_as(m2)
                    loss += (mask1 * m1).sum() / (EPS + mask1.sum())
                    loss += (mask2 * m2).sum() / (EPS + mask2.sum())
                    count += 2
    return coef * loss / count
