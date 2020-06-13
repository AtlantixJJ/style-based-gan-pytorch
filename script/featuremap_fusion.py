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
parser.add_argument("--extractor", default="checkpoint/bedroom_lsun_stylegan_linear_extractor.model")
args = parser.parse_args()

device = "cpu"
colorizer = segviz_torch

generator = model.load_model(args.G)
generator.load_state_dict(torch.load(args.G, map_location=device))
generator.to(device).eval()
torch.manual_seed(3531)
dlatent1 = generator.g_mapping(torch.randn(1, 512).to(device))
dlatent2 = generator.g_mapping(torch.randn(1, 512).to(device))

sep_model = model.semantic_extractor.load_extractor(args.extractor)
sep_model.to(device).eval()


with torch.no_grad():
    image1, stage1 = generator.g_synthesis.mask_latent(
        [torch.ones(1, 1, 1024, 1024)], [dlatent1])
    image2, stage2 = generator.g_synthesis.mask_latent(
        [torch.ones(1, 1, 1024, 1024)], [dlatent2])

image1, image2 = [utils.normalize_image(i)
    for i in [image1, image2]]
label1, label2 = [sep_model(s)[0][:, :336].argmax(1)
    for s in [stage1, stage2]]
label1_viz, label2_viz = [colorizer(l) / 255.
    for l in [label1, label2]]

images = [
    image1, label1_viz,
    image2, label2_viz,]
images = torch.cat(images)
vutils.save_image(images, "source.png", nrow=2)

mask1 = torch.ones(1, 1, 256, 256)
mask1[0, 0, :, :128] = 0
mask2 = 1 - mask1

m_images = []
m_labelvizs = []
with torch.no_grad():
    for st in range(1, dlatent1.shape[1] - 1):
        for i in range(1, st):
            m_images.append(torch.zeros(1, 3, 256, 256))
            m_labelvizs.append(torch.zeros(1, 3, 256, 256))
        for ed in range(st, dlatent1.shape[1]):
            dlatent3 = dlatent1.clone()
            dlatent3[0, st:ed] = dlatent2[0, st:ed]
            image3, stage3 = generator.g_synthesis.mask_latent(
                [mask1, mask2], [dlatent1, dlatent3])
            m_images.append((image3.clamp(-1, 1) + 1) / 2)
            label3 = sep_model(stage3)[0][:, :336].argmax(1)
            label3_viz = colorizer(label3) / 255.
            m_labelvizs.append(label3_viz)
images = torch.cat(m_images + m_labelvizs)
vutils.save_image(images, "manipulation.png", nrow=len(images) // 2)
