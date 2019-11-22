import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
from os.path import join as osj
import glob
from torchvision import utils as vutils
import config
from lib.face_parsing.utils import tensor2label
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="latest", help="log|latest|lerp|evol|seg")
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--zero", type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python script/monitor.py --task %s --model %s --gpu %s --zero %d &"
        basecmd = basecmd % (args.task, osj(args.model, f), args.gpu, args.zero)
        os.system(basecmd)
    exit(0)

savepath = args.model.replace("expr/", "results/")

device = 'cpu'
torch.manual_seed(1314)
latent = torch.randn(1, 512).to(device)
latent.requires_grad = True
noise = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noise.append(torch.randn(1, 1, size, size, device=device))

cfg = config.config_from_name(args.model)
print(cfg)
if 'seg' in args.model:
    from model.tfseg import StyledGenerator
else:
    from model.default import StyledGenerator
generator = StyledGenerator(**cfg).to(device)

if args.zero:
    print("=> Use zero as noise")
    noise = [0] * 18
    for k in range(18):
        size = 4 * 2 ** (k // 2)
        noise[k] = torch.zeros(1, 1, size, size).to(device)
    generator.set_noise(noise)


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
    faceparser = faceparser.to(device)
    faceparser.eval()
    del state_dict

    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        generator.load_state_dict(torch.load(model_file, map_location='cpu'))
        generator.eval()

        gen = generator(latent)
        gen = (gen.clamp(-1, 1) + 1) / 2
        segs = generator.extract_segmentation()
        segs = [s[0].argmax(0) for s in segs]

        with torch.no_grad():
            gen = F.interpolate(gen, 512, mode="bilinear")
            label = faceparser(gen)[0].argmax(0)
            label = utils.idmap(label)
        
        segs += [label]

        segs = [colorizer(s).float() / 255. for s in segs]

        res = segs + [gen[0]]
        res = [F.interpolate(m.unsqueeze(0), 256).cpu()[0] for m in res]
        fpath = savepath + '{}_segmentation.png'.format(i)
        print("=> Write image to %s" % fpath)
        vutils.save_image(res, fpath, nrow=4)