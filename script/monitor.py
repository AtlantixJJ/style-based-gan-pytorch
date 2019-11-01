import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os, glob
from utils import *
from torchvision import utils as vutils
import config
from lib.face_parsing.utils import tensor2label
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="latest", help="log|latest|lerp|evol|seg")
parser.add_argument("--model", default="")
parser.add_argument("--step", type=int, default=7)
parser.add_argument("--lerp", type=float, default=1.0)
args = parser.parse_args()

savepath = args.model.replace("expr/", "results/")

device = 'cuda'
step = args.step
alpha = 1
lerp = args.lerp
shape = 4 * 2 ** step
torch.manual_seed(1314)
latent = torch.randn(1, 512).to(device)
latent.requires_grad = True
noise = []
for i in range(step + 1):
    size = 4 * 2 ** i
    noise.append(torch.randn(1, 1, size, size, device=device))

cfg = config.config_from_name(args.model)
print(cfg)
if 'seg' in args.task:
    from model.seg import StyledGenerator
else:
    from model.default import StyledGenerator
generator = StyledGenerator(512, **cfg).to(device)
model_files = glob.glob(args.model + "/*.model")
model_files.sort()

if "log" in args.task:
    logfile = args.model + "/log.txt"
    with open(logfile) as f:
        head = f.readline().strip().split(" ")
        dic = {h: [] for h in head}
        lines = f.readlines()

    for l in lines:
        items = l.strip().split(" ")
        for h, v in zip(head, items):
            dic[h].append(float(v))

    for k, v in dic.items():
        plt.plot(v)
    plt.legend(list(dic.keys()))
    plt.savefig(savepath + "_loss.png")
    plt.close()

if "seg" in args.task:
    state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
    faceparser = unet.unet()
    faceparser.load_state_dict(state_dict)
    faceparser = faceparser.cuda()
    faceparser.eval()
    del state_dict

    print("=> Load from %s" % model_files[-1])
    generator.load_state_dict(torch.load(
        model_files[-1], map_location='cuda:0'))
    generator.eval()
 
    set_lerp_val(generator.generator.progression, lerp)
    original_generation = generator(latent,
                                noise=noise,
                                step=step,
                                alpha=alpha)
    image = F.interpolate(original_generation, 512, mode="bilinear")
    label = faceparser(image).argmax(1)

    original_generation = normalize_image(original_generation)

    segmentations = generator.generator.extract_segmentation()
    segmentations = segmentations + [label]
    labels = [torch.from_numpy(tensor2label(s[0], s.shape[1]))
        for s in segmentations]
    labels = [l.float().unsqueeze(0) for l in labels]
    res = labels + [original_generation]
    res = [F.interpolate(m, 256).cpu() for m in res]
    res = torch.cat(res, 0)
    print("=> Write image to %s" % (savepath + '_segmentation.png'))
    vutils.save_image(res, savepath + '_segmentation.png', nrow=4)

if "latest" in args.task:
    print("=> Load from %s" % model_files[-1])
    generator.load_state_dict(torch.load(
        model_files[-1], map_location='cuda:0'))
    generator.eval()
    mean_style = generator.mean_style(torch.randn(1024, 512).to(device))

    set_lerp_val(generator.generator.progression, lerp)
    original_generation = generator(latent,
                                    noise=noise,
                                    step=step,
                                    alpha=alpha,
                                    mean_style=mean_style,
                                    style_weight=0.7)
    original_generation = normalize_image(original_generation)

    masks = get_mask(generator.generator.progression)
    masks = [torch.cat([m, m, m], 1) for m in masks]
    res = masks + [original_generation]
    res = [F.interpolate(m, 256) for m in res]
    res = torch.cat(res, 0)
    print("=> Write image to %s" % (savepath + '_latest.png'))
    vutils.save_image(res, savepath + '_latest.png', nrow=cfg['att'])

if "lerp" in args.task:
    generator.load_state_dict(torch.load(
        model_files[-1], map_location='cuda:0'))
    generator.eval()
    mean_style = generator.mean_style(torch.randn(1024, 512).to(device))
    N = 10
    for i in range(N):
        lerp = float(i) / N
        set_lerp_val(generator.generator.progression, lerp)
        original_generation = generator(latent,
                                        noise=noise,
                                        step=step,
                                        alpha=alpha,
                                        mean_style=mean_style,
                                        style_weight=0.7)
        original_generation = normalize_image(original_generation)

        masks = get_mask(generator.generator.progression)
        masks = [torch.cat([m, m, m], 1) for m in masks]
        res = masks + [original_generation]
        res = [F.interpolate(m, 256) for m in res]
        res = torch.cat(res, 0)
        vutils.save_image(res, savepath + '_lerp_%02d.png' %
                          i, nrow=cfg['att'])
    os.system("/usr/bin/ffmpeg -r 2 -f image2 -i {}_lerp_%02d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -b:v 16000k -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -y {}_lerp.mp4".format(savepath, savepath))

if "evol" in args.task:
    for i in range(1, 11):
        generator.eval()
        generator.load_state_dict(torch.load(
            args.model + "/iter_%06d.model" % (i * 1000), map_location='cuda:0'))
        mean_style = generator.mean_style(torch.randn(1024, 512).to(device))
        set_lerp_val(generator.generator.progression, lerp)
        original_generation = generator(latent,
                                        noise=noise,
                                        step=step,
                                        alpha=alpha,
                                        mean_style=mean_style,
                                        style_weight=0.7)
        original_generation = normalize_image(original_generation)

        masks = get_mask(generator.generator.progression)
        masks = [torch.cat([m, m, m], 1) for m in masks]
        res = masks + [original_generation]
        res = [F.interpolate(m, 256) for m in res]
        res = torch.cat(res, 0)
        vutils.save_image(res, savepath + '_evol_%02d.png' %
                          i, nrow=cfg['att'])
    os.system("/usr/bin/ffmpeg -r 2 -f image2 -i {}_evol_%02d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -b:v 16000k -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -y {}_evol.mp4".format(savepath, savepath))
