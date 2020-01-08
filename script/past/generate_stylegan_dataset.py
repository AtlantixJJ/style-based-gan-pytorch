import sys
sys.path.insert(0, '.')
import os, argparse, pickle
import torch
from utils import *
from model.default import StyledGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load", default="checkpoint/stylegan-1024px-new.model")
parser.add_argument("--outdir", default="datasets/stylegan/train/")
args = parser.parse_args()

STEP = 8
ALPHA = 1
N = 10000
torch.manual_seed(1314) #must be it
state_dicts = torch.load(args.load, map_location='cpu')

tg = StyledGenerator(512)
tg.load_state_dict(state_dicts['generator'])
tg.eval()
tg = tg.cuda()

os.system("mkdir %s" % args.outdir)

print("Generating latents")
fpath = args.outdir + "/latents.npz"
if os.path.exists(fpath):
    latents_np = np.load(fpath)['arr_0']
    latents = torch.from_numpy(latents_np)
    del latents_np
else:
    latents = torch.randn(N, 512)
    latents_np = latents.numpy().astype("float32")
    np.savez(fpath, latents_np)
    del latents_np
print("Shape: %s" % str(latents.shape))

noises = []
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noises.append(torch.randn(1, 1, size, size).cuda())

for i in tqdm(range(N)):
    for k in range(STEP + 1):
        noises[k].normal_()
        # too large: 8.1M for a single noise file
        #noises_np = np.empty(len(noises),dtype=object)
        #noises_np[:] = [n.cpu().numpy().astype("float32") for n in noises]
        #fpath = args.outdir + "/latents.npz"
        #np.savez(args.outdir + ("/noises_%05d.npz" % i), noises_np)
        #del noises_np
    target_image = tg(latents[i:i+1].cuda(),
        noise=noises, step=STEP, alpha=ALPHA)
    # (1, 3, 512, 512)
    image = torch2numpy(normalize_image(target_image))[0].transpose(1, 2, 0)
    image = (image * 255).astype("uint8")
    path = args.outdir + ("/%05d.png" % i)
    Image.fromarray(image).save(path, format="PNG")