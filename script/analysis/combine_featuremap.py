import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj

from moviepy.editor import VideoClip

import model, evaluate, utils, config, dataset
from lib.face_parsing import unet
from model.semantic_extractor import get_semantic_extractor, get_extractor_name

from script.analysis.analyze_trace import get_dic
from weight_visualization import concat_weight


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--recursive", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def partial_sum(w, indice, stage, size=256):
    s = []
    for idx in indice:
        stage_ind = cumdims.searchsorted(idx + 1) - 1
        stage_idx = int(idx - cumdims[stage_ind])
        img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
        s.append(img * w[idx])
    if len(s) == 0:
        return torch.zeros(1, 1, size, size, device=device)
    return sum([F.interpolate(a, size=size, mode="bilinear", align_corners=True)
        for a in s])


if args.recursive == "1":
    files = os.listdir(args.model)
    files = [f for f in files if os.path.isdir(f"{args.model}/{f}")]
    models = []
    for i, f in enumerate(files):
        m = glob.glob(f"{args.model}/{f}/*.model")
        if len(m) == 0 or "nonlinear" in m[0] or "generative" in m[0]:
            continue
        models.append(m[0])
    models.sort()
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, f in enumerate(models):
        basecmd = "python script/analysis/featuremap.py --model %s --gpu %s"
        basecmd = basecmd % (f, gpus[i % len(gpus)])
        slots[i % len(gpus)].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)

device = "cpu"
batch_size = 1
latent_size = 512

torch.manual_seed(3)
all_latent = torch.randn(3, 512, device=device)
latent = all_latent[0:1]

model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in args.model else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

fpath = args.model[:args.model.rfind("/")]

generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)
vutils.save_image(image, f"image.png")
dims = [s.shape[1] for s in stage]
cumdims = np.cumsum([0] + dims)

origin_state_dict = torch.load(args.model, map_location=device)
k = list(origin_state_dict.keys())[0]
n_class = origin_state_dict[k].shape[0]
sep_model = get_semantic_extractor(get_extractor_name(args.model))(
    n_class=n_class,
    dims=dims).to(device)
sep_model.load_state_dict(origin_state_dict)

try:
    tw = concat_weight(sep_model.semantic_extractor).unsqueeze(2).unsqueeze(2)
    tw = tw.to(device)
except:
    tw = sep_model.weight.to(device)

if "unit" in args.model:
    tw = F.normalize(tw, 2, 1)
w = tw[:, :, 0, 0].detach().cpu().numpy()

layer_index = 4
feat = stage[layer_index][0]
cname = "nose"
C = utils.CELEBA_CATEGORY.index(cname)
st, ed = cumdims[layer_index:layer_index+2]
label = F.conv2d(feat.unsqueeze(0), tw[:, st:ed])[0].argmax(0)
segmentation = (label == C)
vutils.save_image(segmentation.float(), "segmentation.png")
tw = tw[C, st:ed]
w = w[C, st:ed]

fig = plt.figure(figsize=(9, 4))
x = list(range(w.shape[0]))
plt.scatter(x, w, s=2)
plt.savefig(f"weight_{layer_index}.png")
plt.close()

final_score = (feat * tw).sum(0)
img = final_score / max(-final_score.min(), final_score.max())
viz = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
viz = utils.bu(viz, 256)
vutils.save_image(viz, "result.png")

indice = w.argsort()
w_sorted = w[indice]
zr_ind = w_sorted.searchsorted(0)

nw = w_sorted[:zr_ind]
nw = nw[::-1]
neg_indice = indice[:zr_ind][::-1]
pw = w_sorted[zr_ind:]
pos_indice = indice[zr_ind:]

number = min(nw.shape[0], pw.shape[0])
neg_step = float(nw.shape[0]) / number
pos_step = float(pw.shape[0]) / number

fig = plt.figure(figsize=(9, 8))
ax = plt.subplot(2, 1, 1)
ax.scatter(list(range(nw.shape[0])), nw, s=2, c='r')
ax.scatter(list(range(pw.shape[0])), pw, s=2, c='b')
ax = plt.subplot(2, 1, 2)
px = np.linspace(0, number, pw.shape[0])
nx = np.linspace(0, number, nw.shape[0])
ax.plot(nx, -np.cumsum(nw))
ax.plot(list(range(nw.shape[0])), -np.cumsum(nw))
ax.plot(px, np.cumsum(pw))
plt.savefig(f"pn_{layer_index}.png")
plt.close()

# draw from large to small
def draw_large_small(t):
    i = int(FPS * t)
    if i == 0:
        i = 1
    negs = neg_indice[-int(neg_step * i):].tolist()
    poss = pos_indice[-int(pos_step * i):].tolist()
    inds = negs + poss
    score = (feat[inds] * tw[inds]).sum(0)
    img = score / max(-final_score.min(), final_score.max())#max(-score.min(), score.max())
    viz = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    viz = utils.bu(viz, 256) * 255.
    return utils.torch2numpy(viz.long())[0].transpose(1, 2, 0)

# draw from large to small
def draw_small_large(t):
    i = int(FPS * t)
    negs = neg_indice[:int(neg_step * i)].tolist()
    poss = pos_indice[:int(pos_step * i)].tolist()
    inds = negs + poss
    score = (feat[inds] * tw[inds]).sum(0)
    img = score / max(-final_score.min(), final_score.max()) #max(-score.min(), score.max())
    viz = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    viz = utils.bu(viz, 256) * 255.
    return utils.torch2numpy(viz.long())[0].transpose(1, 2, 0)


def positive_cosine(maps, target):
    score = torch.Tensor(maps.shape[0])
    gt = target > 0
    v2 = target[target > 0].view(-1)
    v2 = v2 / target.norm()
    for i in range(score.shape[0]):
        v1 = maps[i][gt].view(-1)
        v1 = v1 / v1.norm()
        score[i] = torch.dot(v1, v2)
    best_ind = score.argmax()
    return best_ind, score[best_ind]

def full_cosine(maps, target):
    score = torch.Tensor(maps.shape[0])
    gt = target > 0
    v2 = target.view(-1)
    v2 = v2 / target.norm()
    for i in range(score.shape[0]):
        v1 = maps[i].view(-1)
        v1 = v1 / v1.norm()
        score[i] = torch.dot(v1, v2)
    best_ind = score.argmax()
    return best_ind, score[best_ind]

def get_rank_func(rank_func):
    def make_frame(t):
        global prev, curfeat 
        best_ind, sim = rank_func(prev + curfeat, final_score)
        prev = prev + curfeat[best_ind].unsqueeze(0)
        curfeat = torch.cat([curfeat[:best_ind], curfeat[best_ind+1:]])

        #img = prev / max(-prev.min(), prev.max())
        img = prev / max(-final_score.min(), final_score.max())
        viz = utils.heatmap_torch(img.unsqueeze(0))
        viz = utils.bu(viz, 256) * 255.
        return utils.torch2numpy(viz.long())[0].transpose(1, 2, 0)
    
    return make_frame

"""
FPS = 5

animation = VideoClip(draw_large_small, duration=number // FPS)
animation.write_videofile("large_small.mp4", fps=FPS)

animation = VideoClip(draw_small_large, duration=number // FPS)
animation.write_videofile("small_large.mp4", fps=FPS)

FPS = 10

prev = 0
curfeat = feat * tw

animation = VideoClip(
    get_rank_func(positive_cosine),
    duration=feat.shape[0] // FPS)
animation.write_videofile("positive_cosine.mp4", fps=FPS)

prev = 0
curfeat = feat * tw

animation = VideoClip(
    get_rank_func(full_cosine),
    duration=feat.shape[0] // FPS)
animation.write_videofile("full_cosine.mp4", fps=FPS)
"""