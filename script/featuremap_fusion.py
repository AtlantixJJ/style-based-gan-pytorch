import sys
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
import numpy as np
import argparse

import model, utils
from lib.netdissect.segviz import segviz_torch

parser = argparse.ArgumentParser()
parser.add_argument("--G", default="checkpoint/bedroom_lsun_256x256_stylegan.pth")
parser.add_argument("--extractor", default="checkpoint/bedroom_lsun_stylegan_unit_extractor.model")
args = parser.parse_args()

device = "cuda"
colorizer = segviz_torch

generator = model.load_model(args.G)
generator.load_state_dict(torch.load(args.G, map_location=device))
generator.to(device).eval()
torch.manual_seed(3531)
dlatent1 = generator.g_mapping(torch.randn(1, 512).to(device))
dlatent2 = generator.g_mapping(torch.randn(1, 512).to(device))


with torch.no_grad():
    image1, stage1 = generator.g_synthesis.get_stage(dlatent1)
    image2, stage2 = generator.g_synthesis.get_stage(dlatent2)

sep_model = model.semantic_extractor.load_extractor(
    args.extractor,
    dims=[s.shape[1] for s in stage1])
sep_model.to(device).eval()

image1, image2 = [utils.normalize_image(i).cpu()
    for i in [image1, image2]]
label1, label2 = [sep_model(s)[0][0][:, :336].argmax(1)
    for s in [stage1, stage2]]
label1_viz, label2_viz = [colorizer(l) / 255.
    for l in [label1, label2]]

images = [
    image1, label1_viz,
    image2, label2_viz]
images = torch.cat(images)
vutils.save_image(images, "source.png", nrow=2)

mask1 = torch.ones(1, 1, 256, 512, device=device)
mask1[0, 0, :, 256:] = 0
mask2 = 1 - mask1

m_images = []
with torch.no_grad():
    for st in range(dlatent1.shape[1]):
        m_labelvizs = []
        for i in range(st):
            m_images.append(torch.zeros(1, 3, 256, 512))
            m_labelvizs.append(torch.zeros(1, 3, 256, 512))
        for ed in range(st + 1, dlatent1.shape[1] + 1):
            dlatent3 = dlatent1.clone()
            dlatent3[0, st:ed] = dlatent2[0, st:ed]
            image3, stage3 = generator.g_synthesis.mask_latent(
                [mask1, mask2], [dlatent1, dlatent3])
            m_images.append((image3.clamp(-1, 1) + 1) / 2)
            label3 = sep_model(stage3)[0][0][:, :336].argmax(1)
            label3_viz = colorizer(label3) / 255.
            m_labelvizs.append(label3_viz)
        m_images.extend(m_labelvizs)
images = torch.cat([x.cpu() for x in m_images])
vutils.save_image(images,
    "manipulation.png",
    nrow=dlatent1.shape[1],
    padding=5)
