import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import utils

def color_mask(image, color):
    r = image[:, :, 0] == color[0]
    g = image[:, :, 1] == color[1]
    b = image[:, :, 2] == color[2]
    return r & g & b

colors = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),(64, 128, 128),(192, 128, 128)]
fpath = "/Users/jianjinxu/temp/data/20200201_134759_442156_seg.png"

image = np.ascontiguousarray(utils.imread(fpath)[:, :, :3])
h, w, c = image.shape
mark = np.zeros((h, w), dtype="bool")
mask = np.zeros((h + 2, w + 2), dtype="uint8")
count = 1
index = 0
size = h * w

""" # remove small region first
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
for i, c in enumerate(colors):
    m = color_mask(image, c).astype("uint8")
    if m.sum() < 1:
        continue
    m = cv2.dilate(cv2.erode(m, kernel), kernel)
    assert (m & mark).astype("uint8").sum() == 0
    mark |= m.astype("bool")
mark = ~mark
cv2.imwrite("mark.png", mark * 255)
image[mark].fill(0)
"""

# find a False element
def fast_random_seed(mark, val=False, n_retry=1000):
    h, w = mark.shape
    for _ in range(n_retry):
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        if mark[x, y] == val:
            return x, y
    return -1, -1

def slow_random_seed(mark, val=False):
    xs, ys = np.where(mark == val)
    index = np.random.randint(0, len(xs))
    return xs[index], ys[index]

def random_integrated_floodfill(image, n_class=16):
    x = y = 0
    H, W, _ = image.shape
    label = 1 # 0 is ignored label
    mask = np.zeros((H + 2, W + 2), dtype="uint8")
    mark = np.zeros((H, W), dtype="bool")

    mode = 0 # 0: fast; 1: medium; 2: complete

    while not mark.all():
        if mode == 0: # fast random seeding
            x, y = fast_random_seed(mark)
        elif mode == 1: # strict random seeding
            x, y = slow_random_seed(mark)
        # failed to find a seed
        if mark[x, y] and mode == 0:
            mode = 1
            continue
        # flood fill
        mask.fill(0)
        number, _, _, rect = cv2.floodFill(image, mask, (y, x), label, loDiff=0, upDiff=0)
        
        # ignore small region
        if number < 25:
            # use bounding box to reduce memory access
            submask = mask[rect[0]:rect[2], rect[1]:rect[3]].astype("bool")
            image[rect[0]:rect[2], rect[1]:rect[3]][submask].fill(0)
        else:
            label += 1
        # complete markings
        mark |= mask[1:-1,1:-1].astype('bool')
        #mark[rect[0]:rect[2], rect[1]:rect[3]] |= submask
    return image, mark, label


image, mark, label = random_integrated_floodfill(image)
cv2.imwrite("image.png", image)
cv2.imwrite("mark.png", mark * 255)
print(label)