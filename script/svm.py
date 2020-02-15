"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from torchvision import utils as vutils
from lib.face_parsing import unet
import utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg.model")
args = parser.parse_args()

# constants setup
torch.manual_seed(1)
device = 'cpu'

# build model
generator = StyledGenerator(semantic="mul-16-sl2").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)
noise = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noise.append(torch.randn(1, 1, size, size, device=device))
generator.set_noise(noise)
out = generator(latent, seg=False)
out = out.clamp(-1, 1)
vutils.save_image(out, "out.png")

feat_size = max([feat.shape[3] for feat in generator.stage if feat.shape[3] >= 16])
feat = [feat.detach().cpu().numpy()
    for feat in generator.stage if feat.shape[3] >= 16]
del generator 
pickle.dump(feat, open("feat.pkl", "wb"))
# release GPU memory

x = feat[2][0]
x = x.reshape(x.shape[0], -1).transpose()

#feat = feat.reshape(feat.shape[0], -1).transpose()
svm = LinearSVC(fit_intercept=False)
#svm = svm.fit(feat, label.reshape(-1))

