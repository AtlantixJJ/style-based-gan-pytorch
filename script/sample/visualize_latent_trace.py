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
files = files[:4]

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

latents = [np.load(f) for f in files]
#model = PCA()
#model.fit(np.concatenate(latents))
#exp = np.cumsum(model.explained_variance_ratio_)
#print(exp[:5])
for i, f in enumerate(files):
    f = f.replace("_latents.npy", "")
    latent = latents[i]
    model = PCA().fit(latent)
    exp = np.cumsum(model.explained_variance_ratio_)
    cords = model.transform(latent)
    rec = model.inverse_transform(cords)
    diff = np.abs(rec - latents[0]).mean(1)
    s = [int(x) for x in list(diff / diff.min())]
    c = cmap(np.linspace(0, 1, latent.shape[0]))

    plt.scatter(cords[:, 0], cords[:, 1], s=s, c=c)
    plt.savefig(f"{f}_2d.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D {exp[2]:.3f}')
    ax.scatter(cords[:, 0], cords[:, 1], cords[:, 2], s=s, c=c)
    imwrite(f"{f}_3d.png", fig2data(fig))
    azim, elev = ax.azim, ax.elev
    FPS = 24
    number = 240
    def make_frame(t):
        proc = t * FPS / number
        ax.view_init(azim + proc * 30, elev + proc * 360)
        return fig2data(fig)
    plt.close()

    animation = VideoClip(make_frame, duration=number / FPS)
    animation.write_videofile(f"{f}_3droll.mp4", fps=FPS)