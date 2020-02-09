import sys
sys.path.insert(0, ".")
import torch, argparse, tqdm, cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset
import numpy as np
import utils, fid
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="../datasets/CelebAMask-HQ")
parser.add_argument("--output", default="../datasets/CelebAMask-HQ/part_real")
args = parser.parse_args()

ds = dataset.ImageSegmentationDataset(
    root=args.data_dir,
    image_dir="CelebA-HQ-img",
    label_dir="CelebAMask-HQ-mask",
    idmap=utils.idmap,
    random_flip=False)
pds = dataset.ImageSegmentationPartDataset(ds)
dl = DataLoader(pds, batch_size=1, num_workers=2, shuffle=False)


transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def generator():
    for image, label in dl:
        image = utils.tensor2image(image)
        label = utils.torch2numpy(label)[0, 0].astype("uint8")
        parts = utils.image2part(image, label)
        # preprocessing: numpy uint8 -> torch
        for i in range(len(parts)):
            parts[i][1] = transform(Image.fromarray(parts[i][1]))
        yield parts

evaluator = fid.PartFIDEvaluator()
#evaluator.calculate_statistics_given_iterator(generator(), args.output)
evaluator.large_calc_statistic(dl, args.output)
