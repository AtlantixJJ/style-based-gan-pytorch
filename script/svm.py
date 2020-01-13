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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt")
args = parser.parse_args()

# constants setup
torch.manual_seed(1)
device = 'cuda'
imsize = 512
faceparser_path = f"checkpoint/faceparse_unet_{imsize}.pth"

state_dict = torch.load(faceparser_path, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()

# build model
generator = StyledGenerator(semantic="mul-16").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)
out = generator(latent, seg=False)
out = out.clamp(-1, 1)

face_in = F.interpolate(out, imsize, mode="bilinear")
label_logit = faceparser(face_in)
label_logit = F.interpolate(label_logit, out.size(3), mode="bilinear")[0]
label = label_logit.argmax(0)
label = utils.idmap(label) # (1024, 1024)
label = utils.torch2numpy(label)
del faceparser

feat_size = max([feat.shape[3] for feat in generator.stage if feat.shape[3] >= 16])
feat = [F.interpolate(feat.cpu(), feat_size)[0].detach().numpy()
    for feat in generator.stage if feat.shape[3] >= 16]
feat = np.concatenate(feat)
del generator # release GPU memory

feat = feat.reshape(feat.shape[0], -1).transpose()

svm = LinearSVC(penalty="l2", loss="hinge")
svm = svm.fit(feat, label.reshape(-1))

