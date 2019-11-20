import sys
sys.path.insert(0, ".")
import os
os.environ["MKL_NUM_THREADS"]       = "8"
os.environ["NUMEXPR_NUM_THREADS"]   = "8"
os.environ["OMP_NUM_THREADS"]       = "8"
import pickle
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
import numpy as np
from PIL import Image
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
import utils
import lib

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt")
args = parser.parse_args()

# constants setup
torch.manual_seed(1)
device = 'cuda'

# build model
generator = StyledGenerator().to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

# set up input
noise = [[] for i in range(32)]
znoise = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    znoise.append(torch.zeros(1, 1, size, size, device=device))
    for j in range(32):
        noise[j].append(torch.randn(1, 1, size, size, device=device))
latent = torch.randn(1, 512).to(device)

print("=> Generate zero noise features")
generator.set_noise(znoise)
with torch.no_grad():
    out = generator(latent)
    out = (out.clamp(-1, 1) + 1) / 2
vutils.save_image(out, "zero_noise.png")
feat = generator.g_synthesis.stage[6]
np.save("zerofeats.npy", utils.torch2numpy(feat))

print("=> Generate mean nosie features")
feats = []
for i in range(32):
    generator.set_noise(noise[i])
    with torch.no_grad():
        out = generator(latent)
        out = (out.clamp(-1, 1) + 1) / 2
    vutils.save_image(out, "random_noise_%d.png" % i)
    feat = generator.g_synthesis.stage[6]
    feats.append(utils.torch2numpy(feat))
feats = np.concatenate(feats)
np.save("feats.npy", feats)
del generator # release GPU memory

feat_cases = [feats[0:1].mean(0), feats[0:4].mean(0), feats[0:16].mean(0), feats.mean(0)]

for i, X in enumerate(feat_cases):
    C, H, W = X.shape
    X = X.reshape(C, H * W).transpose(1, 0)
    cluster_alg = lib.rcc.RccCluster()
    labels = cluster_alg.fit(X)
    n_labels = labels.max() + 1
    label_map = labels.reshape(H, W)
    label_viz = utils.numpy2label(label_map, n_labels)
    utils.imwrite("rcc_%d.png" % i, label_viz)
    pickle.dump(cluster_alg, open("rcc_cluster_%d.pkl" % i, "wb"))

"""
N = len(feat_list)
print("=> Feature list:")
for i in range(N):
    print("=> [%d] %s: %s" % (i, feat_list[i][0], str(feat_list[i][1].shape)))
    feat_list[i][1] = feat_list[i][1].detach().cpu()

images = []
for i in range(8, N):
    print("=> Clusering [%d] %s" % (i, feat_list[i][0]))
    X = feat_list[i][1][0].numpy() # [512, H, W]
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
"""

""" Test bilinear resize the feature map: not influencing final much. Image is blur too.
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
    vutils.save_image(out2, 'sample%d.png' %
                     i, nrow=1, normalize=True, range=(-1, 1))
    out3 = F.interpolate(out2, basesize, mode='bilinear')
    outs.append(out3)
res = torch.cat(outs, 0)
vutils.save_image(res, 'sample.png', nrow=4, normalize=True, range=(-1, 1))
"""
