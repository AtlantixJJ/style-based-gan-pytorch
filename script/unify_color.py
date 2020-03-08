import sys, glob
sys.path.insert(0, ".")
import numpy as np
import utils
import copy

idx = 3

def find(l, key):
    for i, c in enumerate(l):
        if (key == c).all():
            return i
    return -1

data_dir = sys.argv[1]

mask = np.zeros((512, 512)).astype("bool")

prev = 0
colors = []
prev_region = 0
for k in range(2, 16):
    print(f"=> {k}")
    img = utils.imread(f"{data_dir}/skm_{idx}_{k}.png")
    mask.fill(True)
    region = []
    # update color
    prev = 0
    while mask.sum() > 0:
        for idx in range(prev, 512 * 512):
            x, y = idx // 512, idx % 512
            if mask[x, y]:
                break
        prev = idx
        c = img[x, y]
        cm = utils.color_mask(img, c)
        mask &= ~cm
        region.append([c, cm])
        if find(colors, c) < 0:
            colors.append(c)
    
    if prev_region is 0:
        prev_region = copy.deepcopy(region)
        utils.imwrite(f"{data_dir}/new_skm_{idx}_{k}.png", img)
        continue

    # reassign color
    newimg = np.zeros_like(img)
    count = 0
    flags = [0] * len(region)
    # assign old color
    for pc, pm in prev_region:
        overlaps = []
        for i in range(len(region)):
            if flags[i]:
                overlaps.append(0)
            else:
                cr = region[i][1]
                overlaps.append((pm & cr).sum() / (1e-5 + pm.sum()))
        ind = np.argmax(overlaps)
        cr = region[ind][1]
        # Assign the old color to largest component
        flags[ind] = 1
        pm.fill(False)
        region[ind][0] = pc
        newimg[cr] = pc

    # assign new color
    for i in range(len(region)):
        if flags[i]:
            continue
        cr = region[i][1]
        count += 1
        flags[i] = 1
        newimg[cr] = colors[-count]
        region[i][0] = colors[-count]

    prev_region = copy.deepcopy(region)
    utils.imwrite(f"{data_dir}/new_skm_{idx}_{k}.png", newimg)
    