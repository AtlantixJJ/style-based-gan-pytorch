"""
Given mask sample real
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--model", default="checkpoint/faceparse_unet_512.pth", type=str)
parser.add_argument("--G", default="checkpoint/face_celebahq_1024x1024_stylegan.pth", type=str)
parser.add_argument("--outdir", default="results/baseline_real", type=str)
parser.add_argument("--method", default="LL", type=str)
parser.add_argument("--image", default="", type=str)
parser.add_argument("--label", default="", type=str)
parser.add_argument("--resolution", default=1024, type=int)
parser.add_argument("--n-iter", default=1600, type=int)
parser.add_argument("--n-total", default=16, type=int)
parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import model, utils, optim
from segmenter import get_segmenter

WINDOW_SIZE = 100
n_class = 15
device = "cuda"
extractor_path = args.model
colorizer = utils.Colorize(15)
outdir = args.outdir
optimizer = "adam"

# generator
model_path = args.G
generator = model.load_model(model_path)
generator.to(device).eval()
# target image is controled by seed
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
original_latents = torch.randn(args.n_total, 512)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
image = (1 + image) / 2
dims = [s.shape[1] for s in stage]
sep_model = get_segmenter("celebahq", args.model)

orig_image = torch.zeros(1, 3, 256, 256)
if args.image != "":
    image = torch.from_numpy(utils.imread(args.image)).float() / 255.
    image = image.permute(2, 0, 1).unsqueeze(0)
    orig_image = utils.bu(image, 256)
orig_label = torch.from_numpy(utils.imread(args.label)[:, :, 0]).float()
orig_label = F.interpolate(
    orig_label.unsqueeze(0).unsqueeze(0),
    args.resolution, mode="nearest")[0]
orig_label = orig_label.long().to(device)
name = args.label
name = name[name.rfind("/") + 1:name.rfind(".")]

orig_label_viz = colorizer(orig_label) / 255.
orig_mask = torch.ones_like(orig_label)

res = [orig_image, orig_label_viz]
for ind in range(args.n_total):
    x = original_latents[ind:ind+1].to(device)
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
    image, new_label, latent, noises, record, snapshot = optim.sample_given_mask_external(
        model=generator,
        latent=latent,
        noises=noises,
        label_stroke=orig_label,
        label_mask=orig_mask,
        n_iter=args.n_iter,
        sep_model=sep_model,
        method=f"latent-{args.method}-external",
        mapping_network=generator.g_mapping.simple_forward)
    new_label_viz = colorizer(new_label) / 255.
    res.extend([utils.bu(image, 256), utils.bu(new_label_viz, 256)])

    utils.plot_dic(record, "label edit loss", f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_{ind:02d}_loss.png")

    # make snapshot
    print(snapshot.shape)
    snaps = []
    for i in np.linspace(0, snapshot.shape[0] - 1, 8):
        i = int(i)
        with torch.no_grad():
            el = optim.get_el_from_latent(
                snapshot[i:i+1],
                generator.g_mapping.simple_forward,
                f"latent-{args.method}-external")
            image, stage = generator.get_stage(el, layers)
            image = (1 + image.clamp(-1, 1)) / 2
            label = sep_model(stage)[0].argmax(1)
            label_viz = colorizer(label) / 255.
            snaps.extend([image, label_viz])
    snaps = torch.cat([utils.bu(r, 256) for r in snaps])
    vutils.save_image(snaps, f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_{ind:02d}_snapshot.png", nrow=4)

    # save optimization process
    np.save(
        f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_{ind:02d}_latents.npy",
        utils.torch2numpy(snapshot))

res = torch.cat([utils.bu(r, 256).cpu() for r in res[:8*2]])
vutils.save_image(
    res,
    f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_res.png",
    nrow=4)