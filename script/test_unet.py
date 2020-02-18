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
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet

rootdir = "datasets/CelebAMask-HQ/"
ds = dataset.ImageSegmentationDataset(
    root=rootdir,
    size=512,
    image_dir="CelebA-HQ-img",
    label_dir="CelebAMask-HQ-mask")
dl = torch.utils.data.DataLoader(ds, batch_size=4)

state_dict = torch.load("checkpoint/faceparse_unet_512.pth", map_location='cpu')
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
    label = idmap(y)
    for i in range(tar_seg.shape[0]):
        tar_score = evaluator.calc_single(tar_seg[i], y[i])
evaluator.aggregate()
np.save("results/tar_record.npy", evaluator.summarize())