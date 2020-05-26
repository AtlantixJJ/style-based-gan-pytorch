"""
Given mask sample
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import sys, os, argparse, glob
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--model", default="checkpoint/celebahq_stylegan_unit_extractor.model", type=str)
parser.add_argument("--seed", default=65537, type=int)
parser.add_argument("--indir", default="results/mask_sample", type=str)
parser.add_argument("--outdir", default="results/mask_sample", type=str)
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
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
# target image is controled by seed
torch.manual_seed(args.seed)
#latent = torch.randn(1, 512, device=device)
latent = np.load(f"{args.indir}/s{args.seed}_target.npy")
latent = torch.from_numpy(latent).float().to(device)
original_latents = torch.randn(100, 512)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
image = (1 + image) / 2
dims = [s.shape[1] for s in stage]

layers = list(range(9))
if "layer" in extractor_path:
    ind = extractor_path.rfind("layer") + len("layer")
    s = extractor_path[ind:].split("_")[0]
    layers = [int(i) for i in s.split(",")]
    dims = np.array(dims)[layers].tolist()

sep_model = get_semantic_extractor(get_extractor_name(extractor_path))(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(torch.load(extractor_path, map_location="cpu"))
sep_model.eval()

stage = [s for i, s in enumerate(stage) if i in layers]
segs = sep_model(stage)[0]
label = segs.argmax(1)
label_viz = colorizer(label) / 255.
res = [image, label_viz]

files = glob.glob(f"{args.indir}/adam_s{args.seed}_*latents.npy")
files.sort()

for f in tqdm(files):
    data = np.load(f)
    model = PCA(n_components=3).fit(data)
    cord = model.transform(data)
    rec = model.inverse_transform(cord)
    
    snapshot = torch.from_numpy(data)
    f = f.replace("_latents.npy", "")
    noises = generator.generate_noise(device)
    with torch.no_grad():
        generator.set_noise(noises)
        image, stage = generator.get_stage(snapshot[-1:].to(device))
        image = (1 + image.clamp(-1, 1)) / 2
        label = sep_model(stage)[0].argmax(1)
    label_viz = colorizer(label) / 255.
    res.extend([image, label_viz])

    # make snapshot
    snaps = []
    for i in np.linspace(0, snapshot.shape[0] - 1, 8):
        i = int(i)
        with torch.no_grad():
            image, stage = generator.get_stage(snapshot[i:i+1].to(device))
            image = (1 + image.clamp(-1, 1)) / 2
            label = sep_model(stage)[0].argmax(1)
            label_viz = colorizer(label) / 255.
            snaps.extend([image, label_viz])
    snaps = torch.cat([utils.bu(r, 256) for r in snaps])
    vutils.save_image(snaps, f"{f}_snapshot.png", nrow=4)

res = torch.cat([utils.bu(r, 256) for r in res[:16*2]])
vutils.save_image(
    res,
    f"results/mask_sample/adam_s{args.seed}_n1600_k0.0_res.png",
    nrow=4)