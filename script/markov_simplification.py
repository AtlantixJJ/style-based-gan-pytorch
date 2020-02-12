"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj
from sklearn.metrics.pairwise import cosine_similarity
import utils, config
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="log,seg,fastagreement", help="")
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--recursive", default="0")
args = parser.parse_args()
print(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# for models store in expr, write result to results; for others, store in same dir
savepath = args.model.replace("expr/", "results/")

device = 'cuda' if int(args.gpu) > -1 else 'cpu'

cfg = config.config_from_name(args.model)
print(cfg)
from model.tfseg import StyledGenerator
generator = StyledGenerator(**cfg).to(device)
imsize = 512
batch_size = 2
latent_size = 512
faceparser_path = f"checkpoint/faceparse_unet_{imsize}.pth"

utils.set_seed(65537)
latent = torch.randn(1, latent_size).to(device)
latent.requires_grad = True
noise = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noise.append(torch.randn(1, 1, size, size, device=device))

model_files = glob.glob(args.model + "/*.model")
model_files = [m for m in model_files if "disc" not in m]
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

state_dict = torch.load(faceparser_path, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()
del state_dict

colorizer = utils.Colorize(16) #label to rgb
model_file = model_files[-1]
latent = torch.randn(4, latent_size, device=device)[2:3]
state_dict = torch.load(model_file, map_location='cpu')
missed = generator.load_state_dict(state_dict, strict=False)
if len(missed.missing_keys) > 1:
    print(missed)
    exit()
generator.eval()
generator = generator.to(device)

image = generator(latent, seg=False)
image = image.clamp(-1, 1)
unet_seg = faceparser(F.interpolate(image, size=512, mode="bilinear"))
unet_label = utils.idmap(unet_seg.argmax(1))
unet_label_viz = colorizer(unet_label).float() / 255.
image = (1 + image[0]) / 2
segs = generator.extract_segmentation(generator.stage)
final_label_viz = colorizer(segs[-1].argmax(1)).float() / 255.
images = [image, unet_label_viz, final_label_viz]

prev_label = 0
for i, s in enumerate(segs):
    layer_label = F.interpolate(s, size=image.shape[2], mode="bilinear").argmax(1)[0]
    if prev_label is 0:
        prev_label = layer_label
    layer_label_viz = colorizer(layer_label).float() / 255.
    sum_layers = [F.interpolate(x, size=s.shape[2], mode="bilinear")
        for x in segs[:i]]
    sum_layers = sum(sum_layers) + s
    sum_layers = F.interpolate(sum_layers, size=image.shape[2], mode="bilinear")
    sum_label = sum_layers.argmax(1)[0]
    sum_label_viz = colorizer(sum_label).float() / 255.
    diff_label_viz = sum_label_viz.clone()

    for i in range(3):
        diff_label_viz[i, :, :][sum_label == prev_label] = 1
    images.extend([layer_label_viz, sum_label_viz, diff_label_viz])
    prev_label = sum_label
images = [F.interpolate(img.unsqueeze(0), size=256, mode="bilinear") for img in images]
images = torch.cat(images)
print(f"=> Image write to {savepath}_layer-conv.png")
vutils.save_image(images, f"{savepath}_layer-conv.png", nrow=3)
