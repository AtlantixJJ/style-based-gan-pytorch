import os
from os.path import join as osj
import torch
import torch.nn.functional as F
import argparse
import glob
import numpy as np
import utils
import dataset
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet

rootdir = "/home/xujianjin/data/datasets/CelebAMask-HQ/"
ds = dataset.LatentSegmentationDataset(
    latent_dir=rootdir+"dlatent",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

tar_record = {i:[] for i in range(1,19)}
for latent, image, label in ds:
    image = torch.from_numpy(image).float().cuda()
    image = (image.permute(2, 0, 1) - 127.5) / 127.5
    with torch.no_grad():
        image_ = F.interpolate(image.unsqueeze(0), (512, 512))
        tar_seg = faceparser(image_)[0]
        tar_seg = tar_seg.argmax(0).detach().cpu().numpy()
    tar_score = utils.compute_score(tar_seg, label)
    for i in range(0, 19):
        tar_record[i].append(tar_score[i])
tar_record = utils.aggregate(tar_record)
utils.summarize(tar_record)
np.save("tar_record.npy", tar_record)