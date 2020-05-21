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

from moviepy.editor import VideoClip
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys, os, argparse, glob
from sklearn.decomposition import PCA
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=65537, type=int)
parser.add_argument("--indir", default="results/mask_sample", type=str)
parser.add_argument("--outdir", default="results/mask_sample", type=str)
args = parser.parse_args()

files = glob.glob(f"{args.indir}/adam_s{args.seed}_*latents.npy")
files.sort()


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

NUM = 1
latents = np.concatenate([np.load(f)[-NUM:] for f in files])
model = PCA()
model.fit(latents)
cords = model.transform(latents)
exp = np.cumsum(model.explained_variance_ratio_)
print(exp[:5])
#c = cmap(np.linspace(0, 1, 1600))

plt.scatter(cords[:, 0], cords[:, 1], s=2)
plt.savefig(f"endpoint_{args.seed}.png")
plt.close()
