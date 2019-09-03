import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torchvision import utils
import numpy as np
from PIL import Image
from model import StyledGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--att", default=0)
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
latent.requires_grad = True
# set up mask
#mask = torch.from_numpy(open_image("mask.png")).float().to(device)
#mask = mask.unsqueeze(0).unsqueeze(0)
#mask = torch.nn.functional.interpolate(mask, 512)
#mask = (mask - mask.min()) / (mask.max() - mask.min())
mask = torch.zeros(1, 1, 512, 512, dtype=torch.float32).to(device)
mask[0, 0, 80:140, 80:140] = 1

# build model
generator = StyledGenerator(512, att=args.att).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()
# mean style for truncation
mean_style = generator.mean_style(torch.randn(1024, 512).to(device)).detach()

x = generator(latent,
    noise=noise,
    step=step,
    alpha=alpha,
    mean_style=mean_style,
    style_weight=0.7)

# target
y = x.detach() * 1.1

res = [x]
for i in range(4):
    mseloss = (y - x) * mask
    mseloss = mseloss.sum() / mask.sum()
    mseloss.backward()
    latent.data -= LR * latent.grad
    x = generator(latent,
        noise=noise,
        step=step,
        alpha=alpha,
        mean_style=mean_style,
        style_weight=0.7)
    res.append(x)
#diff = (original_generation - modified_generation).abs().sum(1, keepdim=True)
#thr = (diff > diff.mean() + diff.std()).float()

#thr = torch.cat([thr, thr, thr], 1)
mask = torch.cat([mask, mask, mask], 1)
res = torch.cat(res + [mask])

utils.save_image(res, 'sample.png', nrow=4, normalize=True, range=(-1, 1))
