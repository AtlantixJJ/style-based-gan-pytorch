"""
Given mask sample real
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse, glob
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--vgg", default="checkpoint/vgg.weight")
parser.add_argument("--G", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument("--outdir", default="results/invert_baseline", type=str)
parser.add_argument("--method", default="ML", type=str)
parser.add_argument("--imglist", default="", type=str)
parser.add_argument("--resolution", default=1024, type=int)
parser.add_argument("--n-iter", default=1000, type=int)
parser.add_argument("--n-total", default=1, type=int)
#parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import model, utils, optim
from model.semantic_extractor import get_semantic_extractor, get_extractor_name
import segmenter

WINDOW_SIZE = 100
n_class = 15 if "face" in args.G else 361
device = "cuda"
colorizer = utils.Colorize(15)
outdir = args.outdir
optimizer = "adam"

# perceptual model
pm = model.vgg16.VGG16()
pm.load_state_dict(torch.load(args.vgg))
pm.to(device).eval()

# generator
model_path = args.G
generator = model.load_model(model_path)
generator.to(device).eval()
# torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)

lines = open(args.imglist).readlines()
imagefiles = [l.strip().split(" ")[0] for l in lines]
labelfiles = [l.strip().split(" ")[1] for l in lines]

res = []
for i in range(len(imagefiles)):
    x = torch.randn(1, 512, device=device)

    name = imagefiles[i]
    name = name[name.rfind("/")+1:name.rfind(".")]

    image = torch.from_numpy(utils.imread(imagefiles[i]))
    image = image.float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)

    vgg_image = pm.transform_input(image)
    target_feats = [vgg_image] + pm(vgg_image)

    with torch.no_grad():
        EL = generator.g_mapping(x) # (1, 18, 512)
        GL = EL[:, 0:1, :] # (1, 1, 512)
        ML = x.expand(18, -1).unsqueeze(0) # (1, 18, 512)
    
    if "LL" == args.method:
        latent = x
    elif "GL" == args.method:
        latent = GL
    elif "EL" == args.method:
        latent = EL
    elif "ML" == args.method:
        latent = ML
    
    noises = generator.generate_noise()
    image, latent, noises, record, snapshot = optim.reconstruction(
        model=generator,
        latent=latent,
        noises=noises,
        perceptual_model=pm,
        target_feats=feats,
        n_iter=args.n_iter,
        method=f"latent-{args.method}-internal",
        mapping_network=generator.g_mapping.simple_forward)
    res.append(utils.bu(image, 256))

    utils.plot_dic(record, "label edit loss", f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_0_loss.png")

    # make snapshot
    snaps = []
    for i in np.linspace(0, snapshot.shape[0] - 1, 8):
        i = int(i)
        with torch.no_grad():
            el = optim.get_el_from_latent(
                snapshot[i:i+1],
                generator.g_mapping.simple_forward,
                f"latent-{args.method}-internal")
            image = generator(el)
            image = (1 + image.clamp(-1, 1)) / 2
            snaps.append(image)
    snaps = torch.cat([utils.bu(r, 256) for r in snaps])
    vutils.save_image(snaps, f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_0_snapshot.png", nrow=4)

    # save optimization process
    np.save(
        f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_0_latents.npy",
        utils.torch2numpy(snapshot))

    if i == 16:
        t = torch.cat([utils.bu(r, 256).cpu() for r in res[:16]])
        vutils.save_image(
            t,
            f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_res.png",
            nrow=4)
