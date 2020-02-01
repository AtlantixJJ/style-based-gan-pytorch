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
fpath = "/Users/jianjinxu/temp/label.png"

image = np.ascontiguousarray(utils.imread(fpath)[:, :, :3])
h, w, c = image.shape
empty = np.ones((h + 2, w + 2), dtype="bool")
mask = np.zeros((h + 2, w + 2), dtype="uint8")
count = 1
index = 0
size = h * w

kernel = np.ones((11, 11), dtype="uint8")
# remove small region first
for i, c in enumerate(colors):
    m = color_mask(image, c).astype("uint8")
    if m.sum() < 1:
        continue
    cv2.imwrite("orig_%d.png" % i, m * 255)
    m = cv2.erode(m, kernel)
    cv2.imwrite("erode_%d.png" % i, m * 255)


"""
while empty.sum() > 0 and count < 256:
    mask.fill(0)
    while True: # find an unidentified pixel
        i, j = index // h, index % h
        if empty[i, j]:
            break
        index += 1
    

    #print("Seed: %s | empty: %d" % (str(image[i, j]), empty[i, j]))
    cv2.floodFill(image, mask, (j, i), count,
        loDiff=0, upDiff=0, flags=cv2.FLOODFILL_MASK_ONLY)
    empty = empty & (empty ^ mask.astype("bool"))
    if mask[1:-1, 1:-1].sum() < 25: # ignore small region
        image[mask, 0] = image[mask, 1] = image[mask, 2] = 0
        count -= 1
    else:
        image[mask, 0] = image[mask, 1] = image[mask, 2] = count
        cv2.imwrite("empty%d.png" % count, empty * 255)
        print(f"=> Count {count+1} Left: {empty.sum()}")

    #print(i, j, mask[i,j], mask.sum(), empty.sum())
    #
    count += 1
cv2.imwrite("a.png", image)
cv2.imwrite("b.png", empty * 255)
"""