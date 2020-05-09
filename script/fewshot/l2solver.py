"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--train-size", default=256, type=int)
parser.add_argument(
    "--layer-index", default="4", type=str)
parser.add_argument(
    "--repeat-idx", default=0, type=int)
parser.add_argument(
    "--total-class", default=15, type=int)
parser.add_argument(
    "--debug", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, model
from model.semantic_extractor import get_semantic_extractor, get_extractor_name 

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"
ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=args.data_dir + "/image",
    seg_dir=args.data_dir + "/label")
dl = torch.utils.data.DataLoader(ds, batch_size=args.train_size, shuffle=False)

print("=> Setup generator")
extractor_path = "record/celebahq1/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos-1_l1dev-1_l1unit-1/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

latent = torch.randn(1, 512, device=device)
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
vutils.save_image(image_small[:16], "image.png", nrow=4)
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

# setup test
print("=> Setup test data")
# set up input
layer_index = int(args.layer_index)
colorizer = utils.Colorize(args.total_class)


def get_feature(generator, latent, noise, layer_index):
    with torch.no_grad():
        generator.set_noise(generator.parse_noise(noise))
        image, stage = generator.get_stage(latent)
    feat = stage[layer_index].detach().cpu()
    return image, feat


for ind, sample in enumerate(dl):
    if ind == args.repeat_idx:
        break

latents, noises, images, labels = sample
latents = latents.squeeze(1)
labels = labels[:, :, :, 0].long().unsqueeze(1)
labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels]
fpath = f"results/svm_image_{ind}_b{args.train_size}.png"
images = images.permute(0, 3, 1, 2).float() / 255.
small_images = utils.bu(images[:16], 256)
vutils.save_image(small_images, fpath, nrow=4)

feats = []
for i in tqdm(range(latents.shape[0])):
    image, feat = get_feature(
        generator,
        latents[i:i+1].to(device),
        noises[i].to(device),
        layer_index)
    feats.append(feat)
feats = torch.cat(feats)


print(f"=> Feature shape: {feats.shape}")
print(f"=> Label shape: {labels.shape}")
N, C, H, W = feats.shape
feats = feats.permute(0, 2, 3, 1).reshape(-1, C).float().cpu().numpy()
print(f"=> Feature shape for L2 solver: {feats.shape}")
labels = F.interpolate(labels.float(), size=H, mode="nearest").long().numpy()

print("=> Calculating solver matrix")
XtX = np.matmul(feats.transpose(1, 0), feats)
try:
    XtX_1 = np.linalg.inv(XtX)
except np.linalg.LinAlgError:
    print("=> Failed to solve inversion")
    exit(-1)
solver_mat = np.matmul(XtX_1, feats.transpose(1, 0))

coefs = []
intercepts = []
svs = []
segs = []
cur = 0
for C in range(args.total_class):
    labels_C = labels.copy().reshape(-1, 1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = -1
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")

    coef = np.matmul(solver_mat, labels_C)

    # save result
    coefs.append(coef.transpose(1, 0))
    intercepts.append(0)

    # train result
    est = np.matmul(feats, coef)
    l2 = (est - labels_C).std()
    pred = est > 0
    label = labels_C > 0
    iou = (pred & label).sum().astype("float32") / (pred | label).sum()
    print(f"=> L2 distance: {l2:.3f} \t IoU: {iou:.3f}")
    pred_viz, label_viz = [colorizer(torch.from_numpy(img.reshape(N, 1, H, W))).float() / 255. for img in [pred, label]]

    res = []
    count = 0
    for i in range(8):
        res.extend([label_viz[i:i+1], pred_viz[i:i+1]])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest")
        for r in res]
    fpath = f"results/l2solver_c{C}_l{args.layer_index}_b{args.train_size}.png"
    vutils.save_image(torch.cat(res), fpath, nrow=4)

model_path = f"results/l2solver_l{args.layer_index}_b{args.train_size}.model"
coefs = np.concatenate(coefs)
np.save(model_path, [coefs, 0])