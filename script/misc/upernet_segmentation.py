import sys, argparse, glob
sys.path.insert(0, ".")
from segmenter import get_segmenter
import numpy as np
import torch

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", default="datasets/")
args = parser.parse_args()

imagefiles = glob.glob(f"{args.dir}/*.png") + glob.glob(f"{args.dir}/*.jpg")
imagefiles.sort()

external_model = get_segmenter("bedroom")

for f in imagefiles:
    img = utils.imread(f)
    img = torch.from_numpy(img).float() / 127.5 - 1
    img = img.permute(2, 0, 1).unsqueeze(0).cuda()
    label = external_model.segment_batch(img)[0, 0]
    f = f.replace(".png", ".npy").replace(".jpg", ".npy")
    f = f.replace("image", "sv_label")
    np.save(f, utils.torch2numpy(label))