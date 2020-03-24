import sys
sys.path.insert(0, ".")
import torch, argparse, tqdm, cv2
from torch.utils.data import DataLoader
from dataset import ImageSegmentationDataset
import numpy as np
import utils


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

    parts = utils.image2part(image, label)
    for i, (c, image) in enumerate(parts):
        utils.imwrite(f"{args.output}/{index}_c{i:02d}_l{c:02d}.png", image)
