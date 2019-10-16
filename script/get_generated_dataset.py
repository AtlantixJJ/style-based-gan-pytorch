import sys
sys.path.insert(0, '.')
import os, argparse, pickle
import torch
from utils import *
from model import StyledGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load", default="checkpoint/stylegan-1024px-new.model")
parser.add_argument("--outdir", default=".")
args = parser.parse_args()

STEP = 7
ALPHA = 1
N = 10000
state_dicts = torch.load(args.load, map_location='cpu')

tg = StyledGenerator(512)
tg.load_state_dict(state_dicts['generator'])
tg.eval()
tg = tg.cuda()

os.system("mkdir %s" % args.outdir)

print("Generating latents")
latents = torch.randn(N, 512)
#np.savez(args.outdir + "/latents.npz", latents.numpy())
with open(args.outdir + "/latents.npz", 'wb') as fp:
    pickle.dump(latents.numpy(), fp)

print("Generating noises")
noises = []
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noises.append(torch.randn(N, 1, size, size))
with open(args.outdir + "/noises.npz", 'wb') as fp:
    pickle.dump([n.numpy() for n in noises], fp)

for i in tqdm(range(N)):
    target_image = tg(latents[i:i+1].cuda(), noise=[n[i:i+1].cuda() for n in noises],
        step=STEP, alpha=ALPHA)
    # (1, 3, 512, 512)
    image = torch2numpy(normalize_image(target_image))[0].transpose(1, 2, 0)
    path = args.outdir + ("/%05d.jpg" % i)
    Image.fromarray(image).save(path, format="JPEG")