"""
Given mask sample
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--model", default="checkpoint/celebahq_stylegan_unit_extractor.model", type=str)
parser.add_argument("--outdir", default="results/mask_sample", type=str)
parser.add_argument("--n-iter", default=1000, type=int)
parser.add_argument("--n-total", default=7, type=int)
parser.add_argument("--kl-coef", default=0, type=float)
parser.add_argument("--seed", default=65537, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import model, utils, optim
from model.semantic_extractor import get_semantic_extractor, get_extractor_name

WINDOW_SIZE = 100
n_class = 15
device = "cuda"
extractor_path = args.model
colorizer = utils.Colorize(15)
outdir = args.outdir
optimizer = "adam"

# generator
model_path = "checkpoint/face_ffhq_1024x1024_stylegan2.pth" if "ffhq" in extractor_path else "checkpoint/face_celebahq_1024x1024_stylegan.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
# target image is controled by seed
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
np.save(
    f"{outdir}/s{args.seed}_target.npy",
    utils.torch2numpy(latent))
original_latents = torch.randn(args.n_total, 512)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
image = (1 + image) / 2
dims = [s.shape[1] for s in stage]

layers = list(range(9))
sep_model = 0
if "layer" in extractor_path:
    ind = extractor_path.rfind("layer") + len("layer")
    s = extractor_path[ind:].split("_")[0]
    if ".model" in s:
        s = s.split(".")[0]
    layers = [int(i) for i in s.split(",")]
    dims = np.array(dims)[layers].tolist()

sep_model = get_semantic_extractor(get_extractor_name(extractor_path))(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(torch.load(extractor_path))
sep_model.eval()

with torch.no_grad():
    orig_image, stage = generator.get_stage(latent)
    stage = [s for i, s in enumerate(stage) if i in layers]
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    segs = sep_model(stage)[0]
    orig_label = segs.argmax(1)
    orig_label_viz = colorizer(orig_label) / 255.
    orig_mask = torch.ones_like(orig_label)

res = [orig_image, orig_label_viz]
for ind in range(args.n_total):
    latent.requires_grad = False
    latent.copy_(original_latents[ind])
    latent.requires_grad = True
    noises = generator.generate_noise()
    image, new_label, latent, noises, record, snapshot = optim.sample_given_mask(
        model=generator,
        layers=layers,
        latent=latent,
        noises=noises,
        label_stroke=orig_label,
        label_mask=orig_mask,
        n_iter=args.n_iter,
        kl_coef=args.kl_coef,
        sep_model=sep_model,
        mapping_network=generator.g_mapping.simple_forward)
    new_label_viz = colorizer(new_label) / 255.
    res.extend([utils.bu(image, 256), utils.bu(new_label_viz, 256)])

    utils.plot_dic(record, "label edit loss", f"{outdir}/{optimizer}_s{args.seed}_n{args.n_iter}_k{args.kl_coef}_{ind:02d}_loss.png")

    # make snapshot
    snaps = []
    for i in np.linspace(0, snapshot.shape[0] - 1, 8):
        i = int(i)
        with torch.no_grad():
            image, stage = generator.get_stage(snapshot[i:i+1].cuda(), layers)
            image = (1 + image.clamp(-1, 1)) / 2
            label = sep_model(stage)[0].argmax(1)
            label_viz = colorizer(label) / 255.
            snaps.extend([image, label_viz])
    snaps = torch.cat([utils.bu(r, 256) for r in snaps])
    vutils.save_image(snaps, f"{outdir}/{optimizer}_s{args.seed}_n{args.n_iter}_k{args.kl_coef}_{ind:02d}_snapshot.png", nrow=4)

    # save optimization process
    np.save(
        f"{outdir}/{optimizer}_s{args.seed}_n{args.n_iter}_k{args.kl_coef}_{ind:02d}_latents.npy",
        utils.torch2numpy(snapshot))

res = torch.cat([utils.bu(r, 256) for r in res])
vutils.save_image(
    res,
    f"{outdir}/{optimizer}_s{args.seed}_n{args.n_iter}_k{args.kl_coef}_res.png",
    nrow=4)