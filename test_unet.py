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
    latent_dir=rootdir+"dlatent_test",
    noise_dir=rootdir+"noise_test",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

generator = StyledGenerator()
state_dict = torch.load("checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt", map_location='cpu')
generator.load_state_dict(state_dict, strict=False)
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
noise = None
for i, (latent_np, noise_np, image_np, label_np) in enumerate(ds):
    #image = torch.from_numpy(image).float().cuda()
    #image = (image.permute(2, 0, 1) - 127.5) / 127.5
    latent = torch.from_numpy(latent_np).float().cuda()
    if noise is None:
        noise = [torch.from_numpy(noise).float().cuda() for noise in noise_np]
    else:
        for i in range(len(noise)):
            noise[i] = torch.from_numpy(noise_np[i]).float().cuda()

    with torch.no_grad():
        generator.set_noise(noise)
        image = generator.g_synthesis(latent, step=8).clamp(-1, 1)
        image_ = F.interpolate(image, (512, 512), mode="bilinear")
        tar_seg = faceparser(image_)[0]
        tar_seg = tar_seg.argmax(0).detach().cpu().numpy()
    if evaluator.map_id:
        tar_seg = evaluator.idmap(tar_seg)
        label = evaluator.idmap(label_np)
    tar_score = evaluator.compute_score(tar_seg, label)
    evaluator.accumulate(tar_score)
evaluator.aggregate()
evaluator.summarize()
evaluator.save("tar_record.npy")