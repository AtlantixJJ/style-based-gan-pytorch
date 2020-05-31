import sys, argparse, glob
sys.path.insert(0, ".")
import numpy as np
import torch

import utils
from segmenter import get_segmenter
from lib.netdissect.segviz import segment_visualization_single


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", default="datasets/")
args = parser.parse_args()

imagefiles = glob.glob(f"{args.dir}/*.png") + glob.glob(f"{args.dir}/*.jpg")
imagefiles.sort()

external_model = get_segmenter("bedroom")
colorizer = segment_visualization_single

for f in imagefiles:
    img = utils.imread(f)
    img = torch.from_numpy(img).float() / 127.5 - 1
    img = img.permute(2, 0, 1).unsqueeze(0).cuda()
    label = external_model.segment_batch(img)[0, 0]

    label_viz = utils.torch2numpy(label).astype("uint8")
    label_viz = colorizer(label_viz)

    name = f.replace(".png", ".npy").replace(".jpg", ".npy")
    name = name.replace("image", "sv_label")
    np.save(name, utils.torch2numpy(label))
    utils.imwrite(f.replace("image", "label"), label_viz)
    