import sys
sys.path.insert(0, ".")
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
import numpy as np
from PIL import Image
from model.seg import StyledGenerator
import argparse
import torch.nn.functional as F
from utils import *
import lib
matplotlib.use("agg")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/stylegan-1024px-new.model")
args = parser.parse_args()


def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))


# constants setup
torch.manual_seed(1)
device = 'cuda'
step = 8
alpha = 1
LR = 0.1
shape = 4 * 2 ** step
# set up noise
noise = []
for i in range(step + 1):
    size = 4 * 2 ** i
    noise.append(torch.randn(1, 1, size, size, device=device))
latent = torch.randn(1, 512).to(device)

# cluster
cluster_alg = lib.rcc.RccCluster()

# build model
generator = StyledGenerator(512).to(device)
state_dics = torch.load(args.model, map_location='cpu')
generator.load_state_dict(state_dics['generator'])
generator.eval()
# mean style for truncation
mean_style = generator.mean_style(torch.randn(1024, 512).to(device)).detach()

out1 = generator(latent,
                 noise=noise,
                 step=step,
                 alpha=alpha,
                 mean_style=mean_style,
                 style_weight=0.7)
out1 = out1.detach().cpu()

images = []
for i in range(3, step + 1):
    blk = generator.generator.progression[i]
    X = torch.cat([blk.adain2.gamma, blk.adain2.beta], 1)
    print("=> Clusering [%d] %s" % (i, str(X.shape)))
    C, H, W = X.shape
    X = X.reshape(C, H * W).transpose(1, 0)

    cluster_alg.fit(X)
    labels, n_labels = cluster_alg.compute_assignment(1)
    label_map = labels.reshape(H, W)
    img = label2rgb(label_map)[:,:,:3]
    img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    images.append(F.interpolate(img_t, 128))
    res = images + [F.interpolate(normalize_image(out1), 128)]
    res = torch.cat(res, 0)
    vutils.save_image(res, 'RCC_%d.png' % i, nrow=4, normalize=True, range=(0, 1))