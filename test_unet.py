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

rootdir = "datasets/CelebAMask-HQ/"
ds = dataset.LatentSegmentationDataset(
    latent_dir=rootdir+"dlatent",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

generator = StyledGenerator()
state_dict = torch.load("checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt", map_location='cpu')
generator.load_state_dict(state_dict)
generator = generator.cuda()
generator.eval()
del state_dict


state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

evaluator = utils.MaskCelebAEval(map_id=True)
tar_record = {i:[] for i in range(0, evaluator.n_class)}
for i, (latent, image, label) in enumerate(ds):
    #image = torch.from_numpy(image).float().cuda()
    #image = (image.permute(2, 0, 1) - 127.5) / 127.5
    latent = torch.from_numpy(latent_np).unsqueeze(0).float().cuda()
    with torch.no_grad():
        image = generator(latent).clamp(-1, 1)
        image_ = F.interpolate(image.unsqueeze(0), (512, 512))
        tar_seg = faceparser(image_)[0]
        tar_seg = tar_seg.argmax(0).detach().cpu().numpy()
    if evaluator.map_id:
        tar_seg = evaluator.idmap(tar_seg)
        label = evaluator.idmap(label)
    tar_score = evaluator.compute_score(tar_seg, label)
    evaluator.accumulate(tar_score)
evaluator.aggregate()
evaluator.summarize()
evaluator.save("tar_record.npy")