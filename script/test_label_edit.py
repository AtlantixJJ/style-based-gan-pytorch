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
parser.add_argument("--data-dir", default="data")
parser.add_argument("--seg-cfg", default="mul-16")
parser.add_argument("--lr", default=1e-2, type=int)
parser.add_argument("--n-iter", default=10, type=int)
parser.add_argument("--n-reg", default=3, type=int)
parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()


# constants setup
torch.manual_seed(args.seed)
device = 'cuda'

# build model
generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model, map_location=device))
generator.eval()

ds = dataset.CollectedDataset(args.data_dir)
print(str(ds))
dl = DataLoader(ds, batch_size=1, pin_memory=True)
colorizer = utils.Colorize(16)

for ind, dic in enumerate(dl):
    for k in dic.keys():
        dic[k] = dic[k].to(device)

    latent_ = dic["origin_latent"]
    noises_ = utils.parse_noise(dic["origin_noise"][0])
    label_stroke = dic["label_stroke"]
    label_mask = dic["label_mask"]

    # get original images
    with torch.no_grad():
        extended_latent_ = generator.g_mapping(latent_).detach()
        generalized_latent_ = extended_latent_[:, 0:1, :].detach()
        orig_image, orig_seg = generator(extended_latent_)
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    orig_label_viz = colorizer(orig_label[0]).unsqueeze(0) / 255.
    label_stroke_viz = colorizer(label_stroke[0]).unsqueeze(0) / 255.
    padding_image = torch.zeros_like(orig_image).fill_(-1)
    images = [orig_image, orig_label_viz, label_stroke_viz, orig_label_viz]
    image, label, latent, noises, record = optim.extended_latent_edit_label_stroke(
        model=generator,
        latent=extended_latent_,
        noises=noises_,
        label_stroke=label_stroke,
        label_mask=label_mask,
        lr=args.lr,
        n_iter=args.n_iter)

    label_viz = colorizer(label[0]).unsqueeze(0) / 255.
    diff_image = (orig_image - image).abs().sum(1, keepdim=True)
    diff_image_viz = utils.heatmap_torch(diff_image / diff_image.max())
    diff_label = label_viz.clone()
    for i in range(3):
        diff_label[:, i, :, :][label == orig_label] = 1
    images.extend([image, label_viz, diff_image_viz, diff_label])
    for i in range(len(images)):
        images[i] = images[i].detach().cpu()
    utils.plot_dic(record, "label", f"{args.output}/edit_{ind:02d}_label.png")

    images = torch.cat(images)
    images = F.interpolate(images, (256, 256), mode="bilinear")
    vutils.save_image(images,f"{args.output}/edit_{ind:02d}_result.png", nrow=4)

