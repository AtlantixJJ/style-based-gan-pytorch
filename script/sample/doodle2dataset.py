import sys
sys.path.insert(0, ".")
import utils
import glob
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

indir = sys.argv[1]
outdir = sys.argv[2]
files = glob.glob(f"{indir}/doodle*.png")
files.sort()
colorizer = utils.Colorize(15)

for i, f in enumerate(files):
    img = utils.imread(f)
    mask = torch.from_numpy((img[:, :, 3] < 250).astype("uint8"))
    img = torch.from_numpy(img[:, :, :3]).permute(2, 0, 1)
    label = utils.celeba_rgb2label(img)
    label[mask] = -1
    np.save(f"{outdir}/{i}.npy", label)
    
    label[mask] = 0
    label_viz = colorizer(label) / 255.
    vutils.save_image(
        label_viz.unsqueeze(0),
        f"{outdir}/labelviz_{i}.png")
