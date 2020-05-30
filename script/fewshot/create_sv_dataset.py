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
from lib.netdissect.segviz import segment_visualization, segment_visualization_single

parser = argparse.ArgumentParser()
parser.add_argument(
    "--number", default=32, type=int)
parser.add_argument(
    "--full", default=0, type=int)
parser.add_argument(
    "--out", default="datasets/")
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--task", default="celebahq")
parser.add_argument(
    "--seed", default=65537, type=int) # 1314 for test
args = parser.parse_args()


device = 'cuda'
model_path = args.model

generator = model.load_model(model_path)
generator.to(device).eval()
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
dims = [s.shape[1] for s in stage]
print(dims)
external_model = get_segmenter(
    args.task,
    "checkpoint/faceparse_unet_512.pth")
n_class = len(external_model.get_label_and_category_names()[0][0])
colorizer = utils.Colorize(n_class)
layers = range(2, len(dims))
with open(f"{args.out}/dims.txt", "w") as f:
    f.write(" ".join([str(l) for l in layers]) + "\n")
    f.write(" ".join([str(d) for d in dims]) + "\n")

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
        label = external_model.segment_batch(image, resize=False)
        if image.shape[3] >= 512:
            label = F.interpolate(label.float(), 512, mode="nearest").long()
        
        try:
            label_viz = colorizer(label) / 255.
        except:
            label = label[:, 0, :, :]
            label_viz = utils.torch2numpy(label[0])
            label_viz = torch.from_numpy(colorizer(label_viz))
            label_viz = label_viz.float().permute(2, 0, 1)
            label_viz = label_viz.unsqueeze(0) / 255.

        stage = [s for i, s in enumerate(stage) if i in layers]
        maxsize = max(s.shape[3] for s in stage)
        maxsize = min(maxsize, 512)
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
    else:
        data = utils.torch2numpy(feat.view(feat.shape[0], -1)).transpose(1, 0).astype("float16")
        labels = utils.torch2numpy(label.view(-1))
    if ind == 0:
        print(data.shape, labels.shape)
    np.save(f"{args.out}/sv_feat{ind}", data)
    np.save(f"{args.out}/sv_label{ind}", labels)
    vutils.save_image((image + 1) / 2, f"{args.out}/image{ind}.png")
    vutils.save_image(label_viz, f"{args.out}/label{ind}.png")
    
