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
parser.add_argument("--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument("--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument("--external", default="external")
parser.add_argument("--output", default="results")
parser.add_argument("--data-dir", default="data_image_edit")
parser.add_argument("--seg-cfg", default="conv-16-1")
parser.add_argument("--lr", default=1e-2, type=int)
parser.add_argument("--n-iter", default=10, type=int)
parser.add_argument("--n-reg", default=3, type=int)
parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()


# constants setup
torch.manual_seed(args.seed)
device = 'cuda'
optim_types = [
    "baseline-LL",
    "celossreg-LL-" + args.external,
    "celossreg-GL-" + args.external,
    "celossreg-EL-" + args.external,
    "celossreg-ML-" + args.external]

# build model
generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()

# external model baseline
external_model = unet.unet()
external_model.load_state_dict(torch.load(args.external_model))
external_model = external_model.to(device)
external_model.eval()

ds = dataset.CollectedDataset(args.data_dir)
print(str(ds))
dl = DataLoader(ds, batch_size=1, pin_memory=True)
colorizer = utils.Colorize(16)
idmap = utils.CelebAIDMap()

for ind, dic in enumerate(dl):
    for k in dic.keys():
        dic[k] = dic[k].to(device)

    latent_ = dic["origin_latent"]
    mix_latent_ = latent_.expand(18, -1).detach().clone()
    noises_ = generator.parse_noise(dic["origin_noise"][0])

    # get original images
    with torch.no_grad():
        extended_latent_ = generator.g_mapping(latent_).detach()
        generalized_latent_ = extended_latent_[:, 0:1, :].detach()
        orig_image, orig_seg = generator(extended_latent_)
        ext_seg = idmap.diff_mapid(external_model(orig_image.clamp(-1, 1)))
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

        if "EL" in t:
            latent = extended_latent_
        elif "GL" in t:
            latent = generalized_latent_
        elif "LL" in t:
            latent = latent_
        elif "ML" in t:
            latent = mix_latent_

        image, label, latent, noises, record = optim.edit_image_stroke(
            model=generator,
            external_model=external_model,
            mapping_network=generator.g_mapping.simple_forward,
            latent=latent,
            noises=noises_,
            image_stroke=dic["image_stroke"],
            image_mask=dic["image_mask"],
            method=t,
            lr=args.lr,
            n_iter=args.n_iter,
            n_reg=args.n_reg)
    
        label_viz = colorizer(label[0]).unsqueeze(0) / 255.
        diff_image = (orig_image - image).abs().sum(1, keepdim=True)
        diff_image_viz = utils.heatmap_torch(diff_image / diff_image.max())
        diff_label = label_viz.clone()
        for i in range(3):
            diff_label[:, i, :, :][label == orig_label] = 1
        images.extend([image, label_viz, diff_image_viz, diff_label])
        for i in range(len(images)):
            images[i] = images[i].detach().cpu()
        utils.plot_dic(record, t, f"{args.output}/image_edit_{args.external}_loss_{ind:02d}_{t}.png")

    images = torch.cat(images)
    images = F.interpolate(images, (256, 256), mode="bilinear")
    vutils.save_image(images,f"{args.output}/image_edit_{args.external}_{ind:02d}_result.png", nrow=4)

