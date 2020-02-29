import sys
sys.path.insert(0, ".")
import os
from os.path import join as osj
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse
import glob
import numpy as np
import utils, evaluate
import dataset
import config, segmenter
from lib.face_parsing import unet

rootdir = "../datasets/CelebAMask-HQ/"
ds = dataset.ImageSegmentationDataset(
        root=rootdir,
        size=512,
        image_dir="CelebA-HQ-img",
        label_dir="CelebAMask-HQ-mask",
        idmap=utils.CelebAIDMap(),
        file_list=f"{rootdir}/test.list")
dl = torch.utils.data.DataLoader(ds, batch_size=4)

state_dict = torch.load(sys.argv[1], map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict
idmap = utils.CelebAIDMap().mapid
evaluator = evaluate.MaskCelebAEval()
for i, (x, y) in tqdm(enumerate(dl)):
    x = x.cuda()
    y = y.detach().cpu().numpy()
    with torch.no_grad():
        tar_seg = faceparser(x)
    tar_seg = tar_seg.argmax(1).detach().cpu().numpy()
    tar_seg = idmap(tar_seg)
    for i in range(tar_seg.shape[0]):
        evaluator.calc_single(tar_seg[i], y[i])
evaluator.aggregate()
np.save("results/unet-512.npy", evaluator.summarize())
