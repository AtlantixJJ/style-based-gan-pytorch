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

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python script/monitor.py --task %s --step %d --lerp %d --model %s"
        basecmd = basecmd % (args.task, args.step, args.lerp, osj(args.model, f))
        os.system(basecmd)
    exit(0)

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
    from model.tfseg import StyledGenerator
else:
    from model.tf import StyledGenerator
generator = StyledGenerator(**cfg).to(device)
model_files = glob.glob(args.model + "/*.model")
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

if "log" in args.task:
    logfile = args.model + "/log.txt"
    dic = parse_log(logfile)
    plot_dic(dic, savepath + "_loss.png")

if "seg" in args.task:
    colorizer = Colorize() #label to rgb

    state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
    faceparser = unet.unet()
    faceparser.load_state_dict(state_dict)
    faceparser = faceparser.cuda()
    faceparser.eval()
    del state_dict

    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        generator.load_state_dict(torch.load(model_file, map_location='cuda:0'))
        generator.eval()

        gen = generator(latent)
        gen = (gen.clamp(-1, 1) + 1) / 2
        segs = generator.extract_segmentation()
        segs = [s[0].argmax(0) for s in segs]

        with torch.no_grad():
            gen = F.interpolate(gen, 512, mode="bilinear")
            label = faceparser(gen)[0].argmax(0)
        
        segs += [label]
        for s in segs:
            print(s.shape, s.max())
        segs = [colorizer(s) / 255. for s in segs]
        segs = [s.permute(2, 0, 1).float() for s in segs]

        res = segs + [gen[0]]
        res = [F.interpolate(m.unsqueeze(0), 256).cpu()[0] for m in res]
        for r in res:
            print(r.shape)
        fpath = savepath + '{}_segmentation.png'.format(i)
        print("=> Write image to %s" % fpath)
        vutils.save_image(res, fpath, nrow=4)

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
