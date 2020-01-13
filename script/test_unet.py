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
import utils
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

evaluator = utils.MaskCelebAEval(map_id=True)
for i, (x, y) in tqdm(enumerate(dl)):
    x = x.cuda()
    y = y.detach().cpu().numpy()
    with torch.no_grad():
        tar_seg = faceparser(x)
    tar_seg = tar_seg.argmax(1).detach().cpu().numpy()
    if evaluator.map_id:
        tar_seg = evaluator.idmap(tar_seg)
        label = evaluator.idmap(y)
    for i in range(tar_seg.shape[0]):
        tar_score = evaluator.compute_score(tar_seg[i], y[i])
        evaluator.accumulate(tar_score)
evaluator.aggregate()
evaluator.summarize()
evaluator.save("tar_record.npy")