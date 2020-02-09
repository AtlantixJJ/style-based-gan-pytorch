import sys
sys.path.insert(0, ".")
import torch, argparse, tqdm, cv2
from torch.utils.data import DataLoader
from dataset import ImageSegmentationDataset
import numpy as np
from skimage import morphology
import utils

def get_bbox_mask(mask):
    w = np.where(mask.max(0))[0]
    h = np.where(mask.max(1))[0]
    xmin, xmax = w.min(), w.max()
    ymin, ymax = h.min(), h.max()
    return xmin, ymin, xmax, ymax


def get_square_mask(mask):
    w = np.where(mask.max(0))[0]
    h = np.where(mask.max(1))[0]
    xmin, xmax = w.min(), w.max() + 1
    ymin, ymax = h.min(), h.max() + 1
    cx = (xmin + xmax) / 2.; cy = (ymin + ymax) / 2.
    dw = xmax - xmin; dh = ymax - ymin
    dd = max(dw, dh)
    d = dd / 2.
    xmin, xmax = int(cx - d), int(cx + d)
    ymin, ymax = int(cy - d), int(cy + d)
    dx = dy = 0
    if xmin < 0:
        dx = -xmin
    elif xmax > mask.shape[1]:
        dx = xmax - mask.shape[1]
    if ymin < 0:
        dy = -ymin
    elif ymax > mask.shape[0]:
        dy = ymax - mask.shape[0]
    xmin += dx; xmax += dx; ymin += dy; ymax += dy
    return xmin, ymin, xmax, ymax


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="../datasets/CelebAMask-HQ")
parser.add_argument("--output", default="../datasets/CelebAMask-HQ/part")
args = parser.parse_args()

ds = ImageSegmentationDataset(
    root=args.data_dir,
    image_dir="CelebA-HQ-img",
    label_dir="CelebAMask-HQ-mask",
    idmap=utils.idmap,
    random_flip=False)
dl = DataLoader(ds, batch_size=1, shuffle=False)

for index, (image, label) in enumerate(tqdm.tqdm(dl)):
    image = utils.tensor2image(image)
    label = utils.torch2numpy(label)[0, 0].astype("uint8")

    #region_map, n_region = utils.random_integrated_floodfill(label.copy())
    conn_label, conn_number = morphology.label(label, connectivity=2, return_num=True)

    for i in range(conn_number): # for each connected component
        mask = conn_label == i
        size = mask.sum()
        c = label[mask][0]
        # ignore background
        # ignore image patch smaller than 15x15
        if size < 225 or c == 0: 
            continue

        xmin, ymin, xmax, ymax = get_square_mask(mask)
        
        dst_img = image[ymin:ymax, xmin:xmax].copy()
        src_img = image[ymin:ymax, xmin:xmax].copy()
        cx = dst_img.shape[1] // 2
        cy = dst_img.shape[0] // 2
        submask = ~mask[ymin:ymax, xmin:xmax]
        outside_mean = src_img[submask].mean(0).astype("uint8")
        src_img[:, :] = outside_mean

        dst_img = cv2.seamlessClone(src_img, dst_img, 255 * submask.astype("uint8"), (cx, cy), cv2.NORMAL_CLONE)
        utils.imwrite(f"{args.output}/{index}_c{i:02d}_l{c:02d}.jpg", dst_img)
