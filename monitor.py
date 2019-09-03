import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np
import torch, os
import glob
from torchvision import utils
from model import StyledGenerator
import matplotlib
matplotlib.use("agg")


def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))


def get_mask(styledblocks):
    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.mask1)
            masks.extend(blk.mask2)
    return masks


def set_lerp(progression, lerp):
    for p in progression:
        p.lerp = lerp


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="latest", help="latest|lerp|evol")
parser.add_argument("--model", default="")
parser.add_argument("--att", type=int, default=0)
parser.add_argument("--lerp", type=float, default=0.5)
args = parser.parse_args()

#ind = args.model.rfind('/')
#logfile = args.model[:ind] + "/log.txt"
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
plt.savefig("loss.png")
plt.close()

device = 'cuda'
step = 7
alpha = 1
lerp = args.lerp
shape = 4 * 2 ** step
torch.manual_seed(1)
latent = torch.randn(1, 512).to(device)
latent.requires_grad = True
noise = []
for i in range(step + 1):
    size = 4 * 2 ** i
    noise.append(torch.randn(1, 1, size, size, device=device))

generator = StyledGenerator(512, att=args.att).to(device)
model_files = glob.glob(args.model + "/*.model")
model_files.sort()

if "latest" in args.task:
    generator.load_state_dict(torch.load(model_files[-1], map_location='cuda:0'))
    generator.eval()
    mean_style = generator.mean_style(torch.randn(1024, 512).to(device))

    set_lerp(generator.generator.progression, lerp)
    original_generation = generator(latent,
                                    noise=noise,
                                    step=step,
                                    alpha=alpha,
                                    mean_style=mean_style,
                                    style_weight=0.7)

    masks = get_mask(generator.generator.progression)
    masks = [torch.cat([m, m, m], 1) for m in masks]
    res = masks + [original_generation]
    res = [torch.nn.functional.interpolate(m, 256) for m in res]
    res = torch.cat(res, 0)
    utils.save_image(res, 'latest.png',
                     nrow=4, normalize=True, range=(-1, 1))

if "lerp" in args.task:
    generator.load_state_dict(torch.load(model_files[-1], map_location='cuda:0'))
    generator.eval()
    mean_style = generator.mean_style(torch.randn(1024, 512).to(device))
    N = 10
    for i in range(N):
        lerp = float(i) / N
        set_lerp(generator.generator.progression, lerp)
        original_generation = generator(latent,
                                        noise=noise,
                                        step=step,
                                        alpha=alpha,
                                        mean_style=mean_style,
                                        style_weight=0.7)

        masks = get_mask(generator.generator.progression)
        masks = [torch.cat([m, m, m], 1) for m in masks]
        res = masks + [original_generation]
        res = [torch.nn.functional.interpolate(m, 256) for m in res]
        res = torch.cat(res, 0)
        utils.save_image(res, 'lerp_%02d.png' % i,
                         nrow=4, normalize=True, range=(-1, 1))
    os.system("/usr/bin/ffmpeg -r 2 -f image2 -i lerp_%02d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -b:v 16000k -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -y lerp.mp4")

if "evol" in args.task:
    for i in range(1, 11):
        generator.eval()
        generator.load_state_dict(torch.load(
            args.model + "/iter_%06d.model" % (i * 1000), map_location='cuda:0'))
        mean_style = generator.mean_style(torch.randn(1024, 512).to(device))
        set_lerp(generator.generator.progression, lerp)
        original_generation = generator(latent,
                                        noise=noise,
                                        step=step,
                                        alpha=alpha,
                                        mean_style=mean_style,
                                        style_weight=0.7)

        masks = get_mask(generator.generator.progression)
        masks = [torch.cat([m, m, m], 1) for m in masks]
        res = masks + [original_generation]
        res = [torch.nn.functional.interpolate(m, 256) for m in res]
        res = torch.cat(res, 0)
        utils.save_image(res, 'evol_%02d.png' % i,
                         nrow=4, normalize=True, range=(-1, 1))
    os.system("/usr/bin/ffmpeg -r 2 -f image2 -i evol_%02d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -b:v 16000k -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -y evol.mp4")
