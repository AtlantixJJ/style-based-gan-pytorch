import sys
sys.path.insert(0, ".")
import numpy as np
import cv2 # cv2 swap H, W
import utils
from PIL import Image


def color_mask(image, color):
    r = image[:, :, 0] == color[0]
    g = image[:, :, 1] == color[1]
    b = image[:, :, 2] == color[2]
    return r & g & b

colors = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),(64, 128, 128),(192, 128, 128)]

image_path = "/Users/jianjinxu/temp/data/20200201_134759_442156_seg.png"
image = np.ascontiguousarray(utils.imread(image_path)[:, :, :3])
stroke_path = "/Users/jianjinxu/temp/data/20200201_134615_855961_image_stroke.png"
stroke = utils.pil_read(stroke_path).convert("RGB")
mask_path = "/Users/jianjinxu/temp/data/20200201_134615_887466_image_mask.png"
stroke_mask = utils.pil_read(mask_path).convert("L")

image, label = utils.random_integrated_floodfill(image)
stroke_mask = np.asarray(stroke_mask.resize(image.shape[:2]), dtype="bool")
stroke = np.asarray(stroke.resize(image.shape[:2]))
l = np.bincount(image[stroke_mask, 0]).argmax()
cv2.imwrite("region.png", (image[:, :, 0] == l) * 255)