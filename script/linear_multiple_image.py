"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, math, argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--last-only", default=1, type=int)
parser.add_argument(
    "--train-size", default=4, type=int)
parser.add_argument(
    "--train-iter", default=1000, type=int)
parser.add_argument(
    "--save-iter", default=100, type=int)
parser.add_argument(
    "--repeat-idx", default=0, type=int)
parser.add_argument(
    "--test-dir", default="datasets/Synthesized_test")
parser.add_argument(
    "--test-size", default=1000, type=int)
parser.add_argument(
    "--total-class", default=16, type=int)
parser.add_argument(
    "--debug", default=0, type=int)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model.tf import StyledGenerator
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, loss
from model.semantic_extractor import LinearSemanticExtractor
import pickle

def fg_bg_idmap(x):
    return utils.idmap(x,
        n=args.total_class, # original label number
        map_from=list(range(2, args.total_class)),
        map_to=[1] * 14)

def hair_face_bg_idmap(x):
    return utils.idmap(x,
        n=args.total_class,
        map_from=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        map_to=  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1  ,0,  0,  0,  0,  0])

def full_idmap(x):
    return x

def bedroom_bed_idmap(x):
    return utils.idmap(x,
        n=args.total_class,
        map_from=[2, 3, 4],
        map_to=[0, 0, 0])

idmap = full_idmap
name = "full"


# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"
ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=None,
    seg_dir=args.data_dir + "/label")
dl = torch.utils.data.DataLoader(ds, batch_size=args.train_size, shuffle=False)

test_ds = dataset.LatentSegmentationDataset(
    latent_dir=args.test_dir + "/latent",
    noise_dir=args.test_dir + "/noise",
    image_dir=None,
    seg_dir=args.test_dir + "/label")
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

# build model
print("=> Setup generator")
resolution = utils.resolution_from_name(args.model)
generator = StyledGenerator(resolution=resolution).to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
print(missing_dict)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)
colorizer = utils.Colorize(args.total_class)
image, stage = generator.get_stage(latent, True)
stylegan_dims = [s.shape[1] for s in stage]
train_iter = min(int(10000 / args.train_size), args.train_iter)

def test(generator, linear_model, test_dl):
    result = []
    evaluator = evaluate.MaskCelebAEval()
    for i, sample in enumerate(tqdm(test_dl)):
        latent, noise, image, label = sample
        label = idmap(label[:, :, :, 0])
        generator.set_noise(generator.parse_noise(noise[0].to(device)))
        image, stage = generator.get_stage(latent[0].to(device), detach=True)
        est_label = linear_model.predict(stage) 
        evaluator.calc_single(est_label, utils.torch2numpy(label))

        if i < 4:
            label_viz = colorizer(label).unsqueeze(0).float() / 255.
            est_label_viz = torch.from_numpy(colorizer(est_label))
            est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
            image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
            result.extend([image, label_viz, est_label_viz])

        if args.debug == 1 and i > 4:
            break
            
        if i >= args.test_size:
            break

    global_dic, class_dic = evaluator.aggregate()
    evaluator.summarize()
    return global_dic, class_dic, result


for ind, sample in enumerate(tqdm(dl)):
    if ind == args.repeat_idx:
        break

latents, noises, images, labels = sample
latents = latents.squeeze(1)
labels = labels[:, :, :, 0].long()
labels = idmap(labels)

# Train a linear model based on train_size samples
test_images = 0
linear_model = LinearSemanticExtractor(
    n_class=args.total_class,
    dims=stylegan_dims).to(device)
for i in tqdm(range(train_iter)):
    # ensure we initialize different noise
    generator.set_noise(None)

    # equivalent to 1 iteration, in case memory is not sufficient
    for j in range(latents.shape[0]):
        image, stage = generator.get_stage(latents[j:j+1].to(device), detach=True)
        segs = linear_model(stage, last_only=args.last_only) # (N, C, H, W)
        segloss = loss.segloss(segs, labels[j:j+1].to(device))
        segloss.backward()
        linear_model.optim.step()
        linear_model.optim.zero_grad()

    if i + 1 == train_iter:
        est_labels = segs[-1].argmax(1)

fpath = f"results/linear_{ind}_b{args.train_size}_idmap-{name}.model"
torch.save(linear_model.state_dict(), fpath)
global_dic, class_dic, test_images = test(generator, linear_model, test_dl)
np.save(fpath.replace(".model", "_global.npy"), global_dic)
np.save(fpath.replace(".model", "_class.npy"), class_dic)

image, stage = generator.get_stage(latents[:4, :].to(device), detach=True)
image = (1 + image.clamp(-1, 1).detach().cpu()) / 2
est_labels = torch.from_numpy(linear_model.predict(stage))
est_labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in est_labels]
labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels[:4]]
res = []
for img, lbl, pred in zip(image, labels_viz, est_labels_viz):
    res.extend([img.unsqueeze(0), lbl, pred])
res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest") for r in res]
fpath = f"results/linear_train_{ind}_b{args.train_size}_idmap-{name}.png"
vutils.save_image(torch.cat(res), fpath, nrow=3)

test_images = [F.interpolate(img.detach().cpu(), size=256, mode="nearest")
    for img in test_images]
fpath = f"results/linear_test_{ind}_b{args.train_size}_idmap-{name}.png"
vutils.save_image(torch.cat(test_images), fpath, nrow=3)

labels = []
latents = []