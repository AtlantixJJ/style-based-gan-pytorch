import sys
sys.path.insert(0, ".")
import os
os.environ["MKL_NUM_THREADS"]		="16"
os.environ["NUMEXPR_NUM_THREADS"]	="16"
os.environ["OMP_NUM_THREADS"]		="16"
import torch, argparse, tqdm
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
parser.add_argument("--mode", default="calc_stats", help="real, gen, calc_stats, calc_fid")
# The none_bilinear doesn't have effect on architecture
parser.add_argument("--seg-cfg", default="mul-16-none_sl0")
args = parser.parse_args()


device = 'cuda'
evaluator = fid.PartFIDEvaluator()


# calculate given images
if "real" in args.mode:
    ds = dataset.ImageSegmentationDataset(
        root=args.data_dir,
        image_dir="CelebA-HQ-img",
        label_dir="CelebAMask-HQ-mask",
        idmap=utils.idmap,
        random_flip=False)
    pds = dataset.ImageSegmentationPartDataset(ds)
    dl = DataLoader(pds, batch_size=1, num_workers=2, shuffle=False)

    evaluator.large_calc_statistic(dl, args.output)
    del ds
    del pds
    del dl

# calculate given a model
if "gen" in args.mode:
    generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
    generator.load_state_dict(torch.load(args.model, map_location=device))
    generator.eval()
    generator.to(device)

    ds = dataset.SegGANDataset(generator, save_path=args.output)
    pds = dataset.ImageSegmentationPartDataset(ds)
    dl = DataLoader(pds, batch_size=1, num_workers=0, shuffle=False)
    evaluator.large_calc_statistic(dl, args.output)

    del generator
    del ds
    del pds
    del dl

# calculate mu & sigma intermediate result
if "calc_stats" in args.mode:
    evaluator.summarize_large_feature(args.data_dir)

if "calc_fid" in args.mode:
    path1 = args.data_dir + "/part_gen_ffhq"
    path2 = args.data_dir + "/part_real"
    evaluator.summarize_large_feature(path1, allow_npy=True)
    evaluator.summarize_large_feature(path2, allow_npy=False)
    fids = evaluator.calculate_statistics_given_path(path1, path2)
    print(fids)

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
