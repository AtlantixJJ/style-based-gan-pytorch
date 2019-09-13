import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torchvision import utils
import numpy as np
from PIL import Image
from model import StyledGenerator
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoint/stylegan-512px-running-180000.model")
args = parser.parse_args()

def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))

# constants setup
torch.manual_seed(1)
device = 'cuda'
step = 7
alpha = 1
LR = 0.1
shape = 4 * 2 ** step
# set up noise
noise = []
for i in range(step + 1):
    size = 4 * 2 ** i
    noise.append(torch.randn(1, 1, size, size, device=device))
latent = torch.randn(1, 512).to(device)

# build model
generator = StyledGenerator(512).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()
# mean style for truncation
mean_style = generator.mean_style(torch.randn(1024, 512).to(device)).detach()

N = 16
out1, feats = generator.dbg(latent,
    noise=noise,
    step=step,
    alpha=alpha,
    mean_style=mean_style,
    style_weight=0.7)
L = len(feats) - 3
feat1 = feats[L]
basesize = feat1.shape[2]
outs = []
for i in range(N):
  newsize = int((1 + float(i) / (N - 1)) * basesize)
  feat2 = F.interpolate(feat1, newsize, mode='bilinear')
  print("Input: ", feat2.shape)
  for j in range(L + 1, len(generator.generator.progression) - 1):
    conv = generator.generator.progression[j]
    #print("Progression layer: ", conv)
    feat2 = conv(feat2,
      generator.styles[0],
      F.interpolate(noise[j], 2 * feat2.shape[3], mode='bilinear'))
    print("Output: ", feat2.shape)
  out2 = generator.generator.to_rgb[-2](feat2)
  utils.save_image(out2, 'sample%d.png'%i, nrow=1, normalize=True, range=(-1, 1))
  out3 = F.interpolate(out2, basesize, mode='bilinear')
  outs.append(out3)
res = torch.cat(outs, 0)
utils.save_image(res, 'sample.png', nrow=4, normalize=True, range=(-1, 1))

