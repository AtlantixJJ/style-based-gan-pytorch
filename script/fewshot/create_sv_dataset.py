"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, argparse, pickle
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import utils as vutils

import model, utils
from segmenter import get_segmenter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--number", default=32, type=int)
parser.add_argument(
    "--full", default=0, type=int)
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--seed", default=65537, type=int) # 1314 for test
args = parser.parse_args()


device = 'cuda'

#extractor_path = "checkpoint/ffhq_stylegan2_linear_extractor.model"
#model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"
model_path = args.model

t = "SV"
if "ffhq" in model_path:
    t = "SV2"

generator = model.load_model(model_path)
generator.to(device).eval()
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
dims = [s.shape[3] for s in stage]

external_model = get_segmenter(
    "celebahq",
    "checkpoint/faceparse_unet_512.pth")
#sep_model = model.semantic_extractor.get_semantic_extractor("unit")(
#    n_class=15,
#    dims=dims)
#sep_model.load_state_dict(torch.load(args.external_model))

def random_select(mask, region, number):
    idx, idy = np.where(region)
    inds = np.random.choice(np.arange(len(idx)),
        size=number,
        replace=False)
    for i in inds:
        mask[idx[i], idy[i]] = True

# setup
feats = []
labels = []
for ind in tqdm(range(args.number)):
    latent.normal_()
    with torch.no_grad():
        image = generator(latent)
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image, resize=False)[0]
        #label = sep_model(stage)[0].argmax(1)
        stage = stage[3:8] # layers 3~7 is useful
        maxsize = max(s.shape[3] for s in stage)
        feat = torch.cat([utils.bu(s, maxsize)[0] for s in stage])

    # get the (approximated) support vectors
    if args.full == 0:
        mask = torch.Tensor(maxsize, maxsize).to(device)
        try:
            mask = mask.bool()
        except:
            mask = mask.byte()

        mask[:-1] = label[:-1] != label[1:] # left - right
        mask[:, :-1] |= label[:, :-1] != label[:, 1:] # top - bottom
        mask[1:] |= mask[:-1] # right - left
        mask[:, 1:] |= mask[:, :-1] # bottom - top
        for C in range(15):
            # every hard class is fully sampled
            if C not in [0, 1, 2, 6, 10, 12]: 
                mask |= label == C
        mask = utils.simple_dilate(mask, 3)
        #mask_viz = mask.float().unsqueeze(0).unsqueeze(0)
        #vutils.save_image(mask_viz, "viz.png")
        #print(mask.sum())

        """
        areas = []
        for C in range(15):
            m = label == C
            m = m & ~mask
            if m.sum() == 0:
                continue
            areas.append([C, 1. / m.sum()])
        s = sum([a[1] for a in areas])
        areas = [[c, a/s] for c, a in areas]
        areas.sort(key=lambda x : x[1])
        """
        data = utils.torch2numpy(feat[:, mask].transpose(1, 0))
        labels = utils.torch2numpy(label[mask])

        np.save(f"datasets/{t}/sv_feat{ind}", data)
        np.save(f"datasets/{t}/sv_label{ind}", labels)
    else:
        data = utils.torch2numpy(feat.view(feat.shape[0], -1)).transpose(1, 0)
        labels = utils.torch2numpy(label.view(-1))

        np.save(f"datasets/{t}_full/sv_feat{ind}", data)
        np.save(f"datasets/{t}_full/sv_label{ind}", labels)
    
