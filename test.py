import os
from os.path import join as osj
import torch
import torch.nn.functional as F
import argparse
import glob
import numpy as np
import utils
import dataset
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet

rootdir = "/home/xujianjin/data/datasets/CelebAMask-HQ/"
ds = dataset.LatentSegmentationDataset(
    latent_dir=rootdir+"dlatent",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--step", type=int, default=8)
args = parser.parse_args()

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python test.py --model %s"
        basecmd = basecmd % osj(args.model, f)
        os.system(basecmd)
    exit(0)

out_prefix = args.model.replace("expr/", "results/")

cfg = config.config_from_name(args.model)
print(cfg)
generator = StyledGenerator(**cfg)
generator = generator.cuda()

model_files = glob.glob(args.model + "/*.model")
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)
print("=> Load from %s" % model_files[-1])
missed = generator.load_state_dict(torch.load(
    model_files[-1], map_location='cuda:0'), strict=True)
print(missed)
generator.eval()

state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

record = {i:[] for i in range(0,18)}
record['sigma'] = []
for latent, image, label in ds:
    latent = torch.from_numpy(latent).unsqueeze(0).float().cuda()
    image = torch.from_numpy(image).float().cuda()
    image = (image.permute(2, 0, 1) - 127.5) / 127.5
    with torch.no_grad():
        gen, seg = generator.predict(latent)
    gen = gen[0]
    seg = seg[0].detach().cpu().numpy()
    score = utils.compute_score(seg, label)
    for i in range(0, 18):
        record[i].append(score[i])
    sigma = torch.sqrt(((gen - image)**2).mean())
    record['sigma'].append(utils.torch2numpy(sigma)[()])

record = utils.aggregate(record)
utils.summarize(record)
np.save(f"{out_prefix}_record.npy", record)