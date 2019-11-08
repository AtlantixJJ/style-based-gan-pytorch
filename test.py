import os
from os.path import join as osj
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
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
parser.add_argument("--gpu", default="0")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

evaluator = utils.MaskCelebAEval(map_id=True)

for i, (latent_np, image_np, label_np) in enumerate(ds):
    latent = torch.from_numpy(latent_np).unsqueeze(0).float().cuda()
    image = torch.from_numpy(image_np).float().cuda()
    image = (image.permute(2, 0, 1) - 127.5) / 127.5
    with torch.no_grad():
        gen, seg = generator.predict(latent)
    if evaluator.map_id:
        label = evaluator.idmap(label_np)
    gen = gen[0]
    seg = seg[0].detach().cpu().numpy()
    score = evaluator.compute_score(seg, label)
    evaluator.accumulate(score)
    
    if i == 0:
        genlabel = torch.from_numpy(utils.tensor2label(seg[0], ds.n_class))
        genlabel = genlabel.float().unsqueeze(0)
        gen = gen.unsqueeze(0)
        res = [image, genlabel, gen]
        vutils.save_image(res, f"{out_prefix}.png")

evaluator.aggregate()
evaluator.summarize()
evaluator.save(f"{out_prefix}_record.npy")