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
generator = StyledGenerator(semantic="conv-16-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
print(missing_dict)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)

colorizer = utils.Colorize(16)

def test(generator, latent, svm, N=3):
    result = []
    generator.set_noise(None)
    for _ in range(N):
        latent.normal_()
        image, seg = generator(latent)
        label = seg.argmax(1).detach().cpu()
        feat = generator.stage[4][0] #[for s in generator.stage if s.shape[3] >= 16]
        size = feat.shape[2]
        feat = feat.view(feat.shape[0], -1).permute(1, 0)
        feat = utils.torch2numpy(feat)
        est_label = svm.predict(feat).reshape(size, size)

        label_viz = colorizer(label).unsqueeze(0).float() / 255.
        est_label_viz = torch.from_numpy(colorizer(est_label))
        est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
        image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
        result.extend([image, label_viz, est_label_viz])
    return result


for ind, sample in enumerate(tqdm(dl)):
    latent, noise, image, label = sample
    latent = latent[0].to(device)
    noise = noise[0].to(device)
    label = label[:, :, :, 0]
    print(latent.shape, noise.shape, label.shape)
    generator.set_noise(utils.parse_noise_stylegan(noise))
    image = generator(latent, seg=False)
    feat = generator.stage[4][0] #[for s in generator.stage if s.shape[3] >= 16]
    size = feat.shape[2]
    feat = feat.view(feat.shape[0], -1).permute(1, 0)
    feat = utils.torch2numpy(feat)
    label = F.interpolate(label.unsqueeze(0).float(), size=size, mode="nearest").long()[0, 0]
    svm = LinearSVC(fit_intercept=False)
    svm.fit(feat, label.reshape(-1))
    est_label = svm.predict(feat).reshape(size, size)

    label_viz = colorizer(label).unsqueeze(0).float() / 255.
    est_label_viz = torch.from_numpy(colorizer(est_label))
    est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
    image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
    images = [image, label_viz, est_label_viz]
    images.extend(test(generator, latent, svm))
    images = [F.interpolate(img, size=256, mode="nearest") for img in images]

    vutils.save_image(torch.cat(images), "result.png", nrow=3)
    break

