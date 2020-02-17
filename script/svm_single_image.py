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
    "--noise-aug", default=4, type=int)
parser.add_argument(
    "--layer-index", default=4, type=int)
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
layer_index = args.layer_index
colorizer = utils.Colorize(16)

test_size = 3
test_latents = torch.randn(test_size, 512)
test_noises = [utils.generate_noise_stylegan() for _ in range(test_size)]

def test(generator, test_latents, test_noises, svm, N):
    result = []
    for i in range(N):
        latent = test_latents[i:i+1]
        noise = [n.to(latent.device) for n in test_noises[i]]
        generator.set_noise(noise)
        image, seg = generator(latent)
        label = seg.argmax(1).detach().cpu()
        feat = generator.stage[layer_index][0] #[for s in generator.stage if s.shape[3] >= 16]
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


def get_feature(generator, noise, args):
    pass


for ind, sample in enumerate(tqdm(dl)):
    latent, noise, image, label = sample
    latent = latent[0].to(device)
    label = label[:, :, :, 0].unsqueeze(0).expand(args.noise_aug, -1, -1, -1)
    image = feats = 0

    if args.noise_aug == 1:
        noise = noise[0].to(device)
        generator.set_noise(utils.parse_noise_stylegan(noise))
        image = generator(latent, seg=False)
        feats = generator.stage[layer_index].detach()
    elif args.noise_aug > 1:
        generator.set_noise(None)
        feats = []
        for _ in range(args.noise_aug):
            image = generator(latent, seg=False)
            feats.append(generator.stage[layer_index].detach())
        feats = torch.cat(feats)

    print(f"=> Feature shape: {feats.shape}")
    print(f"=> Label shape: {label.shape}")
    N, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu()
    print(f"=> Feature for SVM shape: {feats.shape}")
    label = F.interpolate(label.float(), size=H, mode="nearest").long()
    svm = LinearSVC(
        dual=(feats.shape[0] < feats.shape[1]),
        fit_intercept=False)
    svm.fit(feats, label.reshape(-1))
    feat = feats.reshape(N, H * W, C)[-1, :, :]
    est_label = svm.predict(feat).reshape(H, W)

    label_viz = colorizer(label[0, 0]).unsqueeze(0).float() / 255.
    est_label_viz = torch.from_numpy(colorizer(est_label))
    est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
    image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
    images = [image, label_viz, est_label_viz]
    images.extend(test(generator, latent, svm))
    images = [F.interpolate(img, size=256, mode="nearest") for img in images]

    vutils.save_image(torch.cat(images), f"results/svm_l{layer_index}_n{args.noise_aug}_result{ind}.png", nrow=3)
    if ind > 3:
        break

