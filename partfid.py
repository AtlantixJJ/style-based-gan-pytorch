import sys
sys.path.insert(0, ".")
import torch, argparse, tqdm, cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset
import numpy as np
import utils, fid, model
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="../datasets/CelebAMask-HQ")
parser.add_argument("--output", default="../datasets/CelebAMask-HQ/part_real")
parser.add_argument("--model", default="")
# The none_bilinear doesn't have effect on architecture
parser.add_argument("--seg-cfg", default="mul-16-none_sl0")
args = parser.parse_args()


device = 'cuda'
evaluator = fid.PartFIDEvaluator()

# calculate given images
if len(args.model) == 0:
    ds = dataset.ImageSegmentationDataset(
        root=args.data_dir,
        image_dir="CelebA-HQ-img",
        label_dir="CelebAMask-HQ-mask",
        idmap=utils.idmap,
        random_flip=False)
    pds = dataset.ImageSegmentationPartDataset(ds)
    dl = DataLoader(pds, batch_size=1, num_workers=1, shuffle=False)

    evaluator.large_calc_statistic(dl, args.output)
    exit(0)

# calculate given a model
generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model, map_location=device))
generator.eval()
generator.to(device)

ds = dataset.SegGANDataset(generator, save_path=args.output)
pds = dataset.ImageSegmentationPartDataset(ds)
dl = DataLoader(pds, batch_size=1, num_workers=0, shuffle=False)
evaluator.large_calc_statistic(dl, args.output)



"""
def data_generator(): # deprecated
    for image, label in dl:
        image = utils.tensor2image(image)
        label = utils.torch2numpy(label)[0, 0].astype("uint8")
        parts = utils.image2part(image, label)
        # preprocessing: numpy uint8 -> torch
        for i in range(len(parts)):
            parts[i][1] = transform(Image.fromarray(parts[i][1]))
        yield parts
"""