def maskarealoss(styledblocks, target=0.5, coef=1.0):
    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.attention1.mask)
            masks.extend(blk.attention2.mask)
    return float(coef) / len(masks) * sum([(m.mean() - target) ** 2 for m in masks])


def maskdivloss(styledblocks, coef=1):
    md = 0
    count = 0
    for blk in styledblocks:
        if blk.att > 0:
            N = len(blk.attention1.mask)
            for i in range(N):
                for j in range(i+1, N):
                    md += (blk.attention1.mask[i] * blk.attention1.mask[j]).mean()
                    md += (blk.attention2.mask[i] * blk.attention2.mask[j]).mean()
                    count += 2
    return md / count
