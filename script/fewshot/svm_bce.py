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
    "--train-size", default=16, type=int)
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

USE_THUNDER = False

# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"
ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=args.data_dir + "/image",
    seg_dir=args.data_dir + "/label")
dl = torch.utils.data.DataLoader(ds, batch_size=args.train_size, shuffle=False)

# build model
if USE_THUNDER:
    print("=> Use thundersvm")
    from thundersvm import OneClassSVM, SVC
else:
    print("=> Use liblinear (multicore)")
    import lib.liblinear.liblinearutil as svm

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
layer_indice = [int(l) for l in args.layer_index.split(",")]
colorizer = utils.Colorize(args.total_class)


def get_feature(generator, latent, noise):
    with torch.no_grad():
        generator.set_noise(generator.parse_noise(noise))
        image, stage = generator.get_stage(latent)
    feats = stage[layer_indice]
    maxsize = max([f.shape[3] for f in feats])
    feat = torch.cat([utils.bu(f, maxsize).detach().cpu() for f in feats], 1)
    return feat


for ind, sample in enumerate(tqdm(dl)):
    if ind == args.repeat_idx:
        break

latents, noises, images, labels = sample
latents = latents.squeeze(1)
labels = labels[:, :, :, 0].long().unsqueeze(1)
labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels]
fpath = f"results/svm_image_{ind}_b{args.train_size}.png"
images = images.permute(0, 3, 1, 2).float() / 255.
small_images = utils.bu(images, 256)
vutils.save_image(small_images, fpath, nrow=4)

feats = []
for i in range(latents.shape[0]):
    feat = get_feature(
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
print(f"=> Feature for SVM shape: {feats.shape}")
labels = F.interpolate(labels.float(), size=H, mode="nearest").long().numpy()

coefs = []
intercepts = []
svs = []
segs = []
cur = 0
for C in range(args.total_class):
    labels_C = labels.copy().reshape(-1)
    mask1 = labels_C == C
    labels_C[mask1] = 1
    labels_C[~mask1] = 0
    ones_size = mask1.sum()
    others_size = (~mask1).sum()
    print(f"=> Class {C} On: {ones_size} Off: {others_size}")
    
    #feats /= np.linalg.norm(feats, 2, 1, keepdims=True)

    if USE_THUNDER:
        svm_model = SVC(kernel="linear", probability=True)
        svm_model.fit(feats, labels_C)
        coefs.append(svm_model.coef_)
        intercepts.append(svm_model.intercept_)
        sv = svm_model.support_vectors_
        svs.append(sv)
        segs.append((cur, cur + sv.shape[0]))
        cur += sv.shape[0]

        # the save function is a shit
        # svm_model.save_to_file(model_path)
        # est_labels = svm_model.predict(feats)
        # draw support vectors
        # indice = svm_model.support_
        # zeros = np.zeros_like(labels_C)
        # zeros[indice] = 1
    else:
        svm_model = svm.train(labels_C, feats, "-n 32 -s 2 -B -1 -q")
        model_path = f"results/svm_train_{ind}_c{C}_l{args.layer_index}_b{args.train_size}.model"
        svm.save_model(model_path, svm_model)
        est_labels, acc, vals = svm.predict(labels_C, feats, svm_model)
        est_labels = np.array(est_labels)
        coef = np.array(svm_model.get_decfun()[0])

    """
    zeros = zeros.reshape(N, 1, H, W)
    est_labels = est_labels.reshape(N, 1, H, W)
    labels_C = labels_C.reshape(N, 1, H, W)
    
    mask = est_labels > 0
    est_labels[mask] = C if C > 0 else 1
    est_labels[~mask] = 0

    mask = labels_C > 0
    labels_C[mask] = C if C > 0 else 1
    labels_C[~mask] = 0

    est_labels_viz = [
        colorizer(l).unsqueeze(0).float() / 255.
        for l in torch.from_numpy(est_labels)]
    labels_C_viz = [
        colorizer(l).unsqueeze(0).float() / 255.
        for l in torch.from_numpy(labels_C)]
    zeros_viz = [
        colorizer(l).unsqueeze(0).float() / 255.
        for l in torch.from_numpy(zeros)]

    res = []
    count = 0
    for lbl, pred, zero in zip(labels_C_viz, est_labels_viz, zeros_viz):
        count += 1
        if count >= 
        res.extend([lbl, pred, zero])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest")
        for r in res]
    fpath = f"results/svm_train_{ind}_c{C}_l{args.layer_index}_b{args.train_size}.png"
    vutils.save_image(torch.cat(res), fpath, nrow=3)
    """

if USE_THUNDER:
    model_path = f"results/svm_l{args.layer_index}_b{args.train_size}.model"
    coefs = np.concatenate(coefs)
    intercepts = np.array(intercepts)
    svs = np.concatenate(svs)[:, 1:] # The first element is shit
    segs = np.array(segs)
    np.save(model_path, [coefs, intercepts, svs, segs])