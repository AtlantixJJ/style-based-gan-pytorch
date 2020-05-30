"""
Given mask sample
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap("plasma")
import matplotlib.style as style
from mpl_toolkits.mplot3d import Axes3D
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from scipy.spatial.distance import cdist
from moviepy.editor import VideoClip
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys, os, argparse, glob
from sklearn.decomposition import PCA
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import model, utils, optim
from model.semantic_extractor import get_semantic_extractor, get_extractor_name


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image.convert("RGB"))
    return image


def imwrite(fp, x):
    Image.fromarray(x).save(open(fp, "wb"))


parser = argparse.ArgumentParser()
parser.add_argument("--ptn", default="i0") #s65537
parser.add_argument("--indir", default="results/mask_sample_real", type=str)
parser.add_argument("--outdir", default="results/mask_sample_real", type=str)
args = parser.parse_args()

WINDOW_SIZE = 100
n_class = 15
device = "cuda"
extractor_path = "checkpoint/celebahq_stylegan_unit_extractor.model"
colorizer = utils.Colorize(15)
outdir = args.outdir
optimizer = "adam"

files = glob.glob(f"{args.indir}/adam_{args.ptn}_*latents.npy")
files.sort()
origin_latent = torch.randn(1, 512, device=device)
if "s" == args.ptn[:1]:
    np.load(f"{args.indir}/{args.ptn}_target.npy")
    origin_latent = torch.from_numpy(origin_latent).float().to(device)


# generator
model_path = "checkpoint/face_ffhq_1024x1024_stylegan2.pth" if "ffhq" in extractor_path else "checkpoint/face_celebahq_1024x1024_stylegan.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(origin_latent)
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

latents = np.stack([np.load(f)[-1] for f in files[:62]])
model = PCA(n_components=3)
model.fit(latents)
cords = model.transform(latents)
exp = np.cumsum(model.explained_variance_ratio_)
print(exp)

dist = cdist(latents, latents)
for i in range(dist.shape[0]):
    dist[i, i] = 0xffffffff
for i in range(dist.shape[0]):
    dist[i, i] = dist.min() + 1
ndist = (dist - dist.min()) / (dist.max() - dist.min())
ndist = np.expand_dims(ndist, 0)
hmap = (utils.heatmap_numpy(ndist)[0] * 255).astype("uint8")
utils.imwrite("heatmap.png", hmap)

# find the minimum distance
ind = dist.reshape(-1).argmin()
a, b = ind // dist.shape[1], ind % dist.shape[1]
latent1 = latents[a]
latent2 = latents[b]
interps = [a * latent1 + (1 - a) * latent2
    for a in np.linspace(1, 0, 8)]

res = []
for latent in tqdm(interps):
    latent = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    with torch.no_grad():
        image, stage = generator.get_stage(latent, layers)
        image = (image.clamp(-1, 1) + 1) / 2
        label = sep_model(stage)[0].argmax(1)
        label_viz = colorizer(label) / 255.
    res.extend([image, label_viz])
res = torch.cat([utils.bu(r, 256) for r in res])
vutils.save_image(res, "interp.png", nrow=4)

res = []
for latent in tqdm(latents[:16]):
    latent = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    with torch.no_grad():
        image, stage = generator.get_stage(latent, layers)
        image = (image.clamp(-1, 1) + 1) / 2
    res.append(image)
res = torch.cat([utils.bu(r, 256) for r in res])
vutils.save_image(res, "test.png", nrow=4)
