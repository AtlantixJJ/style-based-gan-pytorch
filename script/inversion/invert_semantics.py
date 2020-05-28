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
parser.add_argument("--model", default="checkpoint/celebahq_stylegan_unit_extractor.model", type=str)
parser.add_argument("--G", default="checkpoint/face_celebahq_1024x1024_stylegan.pth", type=str)
parser.add_argument("--outdir", default="results/invert_semantics", type=str)
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
extractor_path = args.model
colorizer = utils.Colorize(15)
outdir = args.outdir
optimizer = "adam"

# generator
model_path = args.G
generator = model.load_model(model_path)
generator.to(device).eval()
# torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
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
    label = torch.from_numpy(utils.imread(labelfiles[i])[:, :, 0])
    label = label.long().unsqueeze(0)

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
    image, new_label, latent, noises, record, snapshot = optim.reconstruction_label(
        model=generator,
        layers=layers,
        latent=latent,
        noises=noises,
        target_image=image.to(device),
        target_label=label.to(device),
        n_iter=args.n_iter,
        sep_model=sep_model,
        method=f"latent-{args.method}-internal",
        mapping_network=generator.g_mapping.simple_forward)
    new_label_viz = colorizer(new_label) / 255.
    res.extend([utils.bu(image, 256), utils.bu(new_label_viz, 256)])

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
            image, stage = generator.get_stage(el, layers)
            image = (1 + image.clamp(-1, 1)) / 2
            seg = sep_model(stage)[0]
            if type(seg) is list:
                seg = seg[0]
            label = seg.argmax(1)
            label_viz = colorizer(label) / 255.
            snaps.extend([image, label_viz])
    snaps = torch.cat([utils.bu(r, 256) for r in snaps])
    vutils.save_image(snaps, f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_0_snapshot.png", nrow=4)

    # save optimization process
    np.save(
        f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_0_latents.npy",
        utils.torch2numpy(snapshot))

    if i == 8:
        t = torch.cat([utils.bu(r, 256).cpu() for r in res[:16]])
        vutils.save_image(
            t,
            f"{outdir}/{optimizer}_i{name}_n{args.n_iter}_m{args.method}_res.png",
            nrow=4)
