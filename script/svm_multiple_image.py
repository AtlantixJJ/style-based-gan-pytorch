"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from torchvision import utils as vutils
from lib.face_parsing import unet
import utils, dataset
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--layer-index", default=4, type=int)
parser.add_argument(
    "--total-class", default=5, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# constants setup
torch.manual_seed(1)
device = "cuda" if int(args.gpu) > -1 else "cpu"
ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=None,
    seg_dir=args.data_dir + "/label")
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# build model
resolution = utils.resolution_from_name(args.model)
generator = StyledGenerator(resolution=resolution, semantic="conv-16-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
print(missing_dict)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)
layer_index = args.layer_index
colorizer = utils.Colorize(args.total_class)

def fg_bg_idmap(x):
    return utils.idmap(x,
        n=16, # original label number
        map_from=list(range(2, 16)),
        map_to=[1] * 14)

def hair_face_bg_idmap(x):
    return utils.idmap(x,
        n=16,
        map_from=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        map_to=  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1  ,0,  0,  0,  0,  0])

def full_idmap(x):
    return x


idmap = full_idmap
name = "fulll"

test_size = 16
test_latents = torch.randn(test_size, 512)
test_noises = [generator.generate_noise() for _ in range(test_size)]

def test(generator, svm, test_latents, test_noises, N):
    result = []
    evaluator = utils.MaskCelebAEval()
    for i in range(N):
        latent = test_latents[i:i+1].to(device)
        noise = [n.to(latent.device) for n in test_noises[i]]
        generator.set_noise(noise)
        image, seg = generator(latent)
        label = seg.argmax(1).detach().cpu()
        label = idmap(label)
        feat = generator.stage[layer_index][0] #[for s in generator.stage if s.shape[3] >= 16]
        size = feat.shape[2]
        feat = feat.view(feat.shape[0], -1).permute(1, 0)
        feat = utils.torch2numpy(feat)
        est_label = svm.predict(feat).reshape(size, size)

        if i < 4:
            label_viz = colorizer(label).unsqueeze(0).float() / 255.
            est_label_viz = torch.from_numpy(colorizer(est_label))
            est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
            image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
            result.extend([image, label_viz, est_label_viz])
    return result


feats = []
labels = []
for ind, sample in enumerate(tqdm(dl)):
    latent, noise, image, label = sample
    latent = latent[0].to(device)
    label = label[:, :, :, 0].unsqueeze(0)
    label = idmap(label)
    labels.append(label)

    image = generator(latent, seg=False)
    feat = generator.stage[layer_index].detach()
    feats.append(feat)
feats = torch.cat(feats)
labels = torch.cat(labels)

print(f"=> Feature shape: {feats.shape}")
print(f"=> Label shape: {labels.shape}")
N, C, H, W = feats.shape
feats = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu()
print(f"=> Feature for SVM shape: {feats.shape}")
labels = F.interpolate(labels.float(), size=H, mode="nearest").long()
svm = LinearSVC(
    dual=(feats.shape[0] < feats.shape[1]),
    fit_intercept=False)
svm.fit(feats, labels.reshape(-1))

images = test(generator, svm, test_latents, test_noises, test_size)
images = [F.interpolate(img, size=256, mode="nearest") for img in images]

vutils.save_image(torch.cat(images), f"results/svm_l{layer_index}_idmap-{name}_result{ind}.png", nrow=3)
