"""
Use cross entropy loss to regularize the edit
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import glob
import optim, model, utils, dataset
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoint/fixseg.model")
parser.add_argument("--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument("--output", default="results")
parser.add_argument("--seg-cfg", default="mul-16")
parser.add_argument("--lr", default=1e-2, type=int)
parser.add_argument("--n-iter", default=200, type=int)
parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()


# constants setup
torch.manual_seed(args.seed)
device = 'cuda'
optim_types = ["baseline-latent","celossregexternal-latent","celossregexternal-latent-slow","celossreg-latent","celossreg-latent-slow"]

# build model
generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()

# external model baseline
external_model = unet.unet()
external_model.load_state_dict(torch.load(args.external_model))
external_model = external_model.to(device)
external_model.eval()

ds = dataset.CollectedDataset("data")
print(str(ds))
dl = DataLoader(ds, batch_size=1, pin_memory=True)
colorizer = utils.Colorize(16)

for ind, dic in enumerate(dl):
    for k in dic.keys():
        dic[k] = dic[k].to(device)

    latent = dic["origin_latent"][0]
    noises = utils.parse_noise(dic["origin_noise"][0])

    # get original images
    with torch.no_grad():
        orig_image, orig_seg = generator(latent)
        ext_seg = utils.diff_idmap(external_model(orig_image.clamp(-1, 1)))
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    ext_label = ext_seg.argmax(1)
    ext_label_viz = colorizer(ext_label[0]).unsqueeze(0) / 255.
    orig_label = orig_seg.argmax(1)
    orig_label_viz = colorizer(orig_label[0]).unsqueeze(0) / 255.
    padding_image = torch.zeros_like(orig_image).fill_(-1)
    image_stroke = orig_image * (1 - dic["image_mask"]) + dic["image_stroke"] * dic["image_mask"]

    images = [orig_image, orig_label_viz, image_stroke, ext_label_viz]

    for t in optim_types:
        print("=> Optimization method %s" % t)
        orig_image, orig_label, image, label, latent, noises, record = optim.get_optim(t,
            external_model=external_model,
            model=generator,
            latent=dic["origin_latent"][0],
            noises=utils.parse_noise(dic["origin_noise"][0]),
            image_stroke=dic["image_stroke"],
            image_mask=dic["image_mask"],
            lr=args.lr,
            n_iter=args.n_iter)
    
        label_viz = colorizer(label[0]).unsqueeze(0) / 255.
        diff_image = (orig_image - image).abs().sum(1, keepdim=True)
        diff_image_viz = utils.heatmap_torch(diff_image / diff_image.max())
        diff_label = label_viz.clone()
        for i in range(3):
            diff_label[:, i, :, :][label == orig_label] = 1
        images.extend([image, label_viz, diff_image_viz, diff_label])
        utils.plot_dic(record,   f"{args.output}/edit_{ind:02d}_{t}.png")

    images = torch.cat(images)
    images = F.interpolate(images, (512, 512), mode="bilinear")
    vutils.save_image(images,f"{args.output}/edit_{ind:02d}_result.png", nrow=4)

