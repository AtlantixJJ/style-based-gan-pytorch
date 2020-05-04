import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj

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

device = "cuda"
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
vutils.save_image(image, f"{fpath}_image.png")
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
    w = tw[:, :, 0, 0].detach().cpu().numpy()
except:
    tw = sep_model.weight.to(device)
    w = tw[:, :, 0, 0].detach().cpu().numpy()
dic, cr, cp, cn = get_dic(w)

print("=> Show statistics of each featuremap")
mins = []
maxs = []
absv = []
for s in stage:
    for i in range(s.shape[1]):
        mins.append(utils.torch2numpy(s[0, i].min()))
        maxs.append(utils.torch2numpy(s[0, i].max()))
        absv.append(utils.torch2numpy(s[0, i].abs().mean()))
x = list(range(len(mins)))
plt.scatter(x, mins, s=2)
plt.scatter(x, maxs, s=2)
plt.scatter(x, absv, s=2)
plt.savefig(f"{fpath}_min.png")
plt.close()



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


print("=> Stage contribution")
# each stage is of the same resolution
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
    image_small = F.interpolate(image,
        size=256, mode="bilinear", align_corners=True)

prev, cur = 0, 0
scores = []
for s in stage:
    cur += s.shape[1]
    score = F.conv2d(s, tw[:, prev:cur])
    scores.append(score)
    prev = cur

for C in [1, 4]:
    cname = utils.CELEBA_CATEGORY[C]
    print(f"=> Category {cname}")

    # individual layer
    sc = [s[:, C:C+1] for s in scores]
    maxi = max([max(-a.min(), a.max()) for a in sc])
    arr = [utils.heatmap_torch(m / maxi) for m in sc]

    # cumulative layers
    cur = sc[0]
    cc = [cur]
    uc = []
    for s in sc[1:]:
        uc.append(utils.bu(cur, s.shape[3]))
        cur = uc[-1] + s
        cc.append(cur)
    cumarr = [a / max(-a.min(), a.max()) for a in cc]
    cumarr = [utils.heatmap_torch(m) for m in cumarr]
    
    # difference is cumulative layers
    dc = [cc[i] - utils.bu(cc[i-1], cc[i].shape[3])
        for i in range(1, len(cc))]
    for i in range(len(dc)):
        if dc[i].shape[3] < 64:
            continue
        #dilated = uc[i] + uc[i].std() * 0.2
        #dc[i][dilated <= 0] = 0
        #diffcum[i][cc[i+1] > 0] = 1
    diffcum = [a / max(-a.min(), a.max()) for a in dc]
    diffcum = [utils.heatmap_torch(m) for m in diffcum]
    line = cc[-1].abs() < cc[-1].std() * 0.2
    for i in range(len(diffcum)):
        if dc[i].shape[3] < 64:
            continue
        cline = F.interpolate(line.float(),
            size=dc[i].shape[3]) > 0
        for j in range(3):
            diffcum[i][:, j:j+1, :, :][cline] = 0 # black
    diffcum = [torch.zeros_like(diffcum[0])] + diffcum
    
    # make a pyramid
    p1 = utils.make_pyramid(arr)
    p2 = utils.make_pyramid(cumarr)
    p3 = utils.make_pyramid(diffcum)

    # save image
    imgs = torch.cat([p1, p2, p3])
    vutils.save_image(imgs, f"{fpath}_{cname}_stage.png", nrow=1)


print("=> Weight distribution")
wsorted = w.copy()
for i in range(wsorted.shape[0]):
    wsorted[i].sort()
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(1, 2, 1)
    ax.plot(wsorted[i])
    ax = plt.subplot(1, 2, 2)
    ax.hist(wsorted[i], bins=100)
    NT, PT = - w[i][w[i]<0].std(), w[i][w[i]>0].std()
    ax.axvline(x=NT)
    ax.axvline(x=PT)
    plt.savefig(f"{fpath}_weight_distribution_{i}.png")
    plt.close()


print("=> Segments of weight")
for C in [1, 4]:
    indice = list(cr[C])
    cname = utils.CELEBA_CATEGORY[C]
    print(f"=> Category {cname} size {len(indice)}")

    NT, PT = - w[C][w[C]<0].std(), w[C][w[C]>0].std()
    negatives = [-10, 3 * NT, 2 * NT, 1 * NT, 0]
    positives = [10, 3 * PT, 2 * PT, 1 * PT, 0]

    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = (image.clamp(-1, 1) + 1) / 2
        image_small = F.interpolate(image, size=256, mode="bilinear", align_corners=True)
    
    arr_pos = [partial_sum(
        w[C],
        np.where((bg <= w[C]) & (w[C] < ed))[0],
        stage) for ed, bg in zip(positives[:-1], positives[1:])]

    arr_neg = [-partial_sum(
        w[C],
        np.where((bg <= w[C]) & (w[C] < ed))[0],
        stage) for bg, ed in zip(negatives[:-1], negatives[1:])]

    arr_total = [a - b for a, b in zip(arr_pos, arr_neg)]

    final = sum(arr_total)
    final = final / max(-final.min(), final.max())
    arr = arr_pos + arr_neg + arr_total
    maxi = max([max(-a.min(), a.max()) for a in arr])
    arr = [a / maxi for a in arr] + [final]  

    imgs = torch.cat([utils.heatmap_torch(m) for m in arr] + [image_small])
    vutils.save_image(imgs, f"{fpath}_{cname}_segments.png", nrow=4)



print("=> Positive and negative")

for C in [1, 4]:
    indice = list(cr[C])
    cname = utils.CELEBA_CATEGORY[C]
    print(f"=> Category {cname} size {len(indice)}")

    NT, PT = - w[C][w[C]<0].std(), w[C][w[C]>0].std()
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = (image.clamp(-1, 1) + 1) / 2
        image_small = F.interpolate(image, size=256, mode="bilinear", align_corners=True)

    mp = partial_sum(w[C], np.where(w[C] > 0)[0], stage)
    mn = -partial_sum(w[C], np.where(w[C] < 0)[0], stage)
    mt = partial_sum(w[C], np.where(w[C] != 0)[0], stage)

    sp = partial_sum(w[C], np.where((0 <= w[C]) & (w[C] < PT))[0], stage)
    sn = -partial_sum(w[C], np.where((NT <= w[C]) & (w[C] < 0))[0], stage)
    lp = partial_sum(w[C], np.where(PT <= w[C])[0], stage)
    ln = -partial_sum(w[C], np.where(w[C] < NT)[0], stage)
    diff_s = sp - sn
    diff_l = lp - ln
    res = sp - sn + lp - ln
    arr = [mp, mn, mt, sp, sn, lp, ln, diff_s, diff_l, res]
    maxi = max([a.max() for a in arr])

    img_p, img_n, img_t, img_sp, img_sn, img_lp, img_ln, imgdiff_s, imgdiff_l, imgres = [
        utils.heatmap_torch(m / max(-m.min(), m.max())) for m in arr]
    
    img = torch.cat([
        img_p, img_n, image_small, img_t,
        img_sp, img_sn, imgdiff_s, imgres,
        img_lp, img_ln, imgdiff_l])
    vutils.save_image(img, f"{fpath}_{cname}_positive_negative.png", nrow=4)



# random projection
"""
print("=> Random projection")
w = torch.randn((sum(dims),))
wp = w.clone(); wp[wp < 0] = 0
f = 0
count = 0
for s in stage:
    for i in range(s.shape[1]):
        m = F.interpolate(s[0:1, i:i+1], size=256, mode="bilinear", align_corners=True)
        f = f + m * w[count]
        count += 1
f = (f - f.min()) / (f.max() - f.min())
img = utils.heatmap_torch(f)
vutils.save_image(img, "random_full.png")
"""


# Selected featuremaps
print("=> Visualize selected featuremaps")
cname = "eye"
C = utils.CELEBA_CATEGORY.index(cname)

print(f"=> Category {cname} size {len(indice)}")

indice = w[C, :].argsort()
indice = indice[:5].tolist() + indice[-5:].tolist()

cname = utils.CELEBA_CATEGORY[C]
print(f"=> Category {cname} size {len(indice)}")

for ind in range(1):
    latent = all_latent[ind:ind+1]
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = (image.clamp(-1, 1) + 1) / 2
        image_small = F.interpolate(image, size=256, mode="bilinear", align_corners=True)

    viz_imgs = []
    for idx in indice:
        stage_ind = cumdims.searchsorted(idx + 1) - 1
        stage_idx = int(idx - cumdims[stage_ind])
        img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
        attr = "-".join(dic[idx])
        viz_imgs.append((img, idx, attr))

    for i, (img, idx, attr) in enumerate(viz_imgs):
        if img.shape[3] >= 1024:
            continue
        img = img / max(-img.min(), img.max())
        if img.shape[3] <= 256:
            img = F.interpolate(img, size=256, mode="bilinear", align_corners=True)
        img = utils.heatmap_torch(img)
        vutils.save_image(img, f"{fpath}_{ind}_{i:03d}_{idx:04d}_{attr}.png")
        vutils.save_image(image_small, f"{ind}_image.png")


# score map
score = sep_model(stage)[0][0]
mini, maxi = score.min(), score.max()
imgs1 = []
imgs2 = []
for i in range(score.shape[0]):
    img = score[i].clone()
    img = (img - mini) / (maxi - mini)
    img = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    imgs1.append(img)

    img = score[i]
    img[img < 0] = 0
    img = img / maxi
    img = utils.heatmap_torch(img.unsqueeze(0).unsqueeze(0))
    imgs2.append(img)

imgs1 = torch.cat([F.interpolate(img, size=256) for img in imgs1])
imgs2 = torch.cat([F.interpolate(img, size=256) for img in imgs2])
vutils.save_image(imgs1, "score_full.png", nrow=4)
vutils.save_image(imgs2, "score_positive.png", nrow=4)




catid = 1

# weight attribution
#w = origin_state_dict["weight"][:, :, 0, 0]
#w = F.normalize(w, 2, 1)
#aw = [w[i].argsort(descending=True) for i in range(w.shape[0])]

# contribution
"""
for i in range(len(stage)):
    stage[i].requires_grad = True

score = []
rank = []
for catid in range(2,3):
    y = sep_model(stage)[0][0, catid, :, :].sum()
    gstage = torch.autograd.grad(y, stage)
    contribution = [stage[i] * gstage[i] for i in range(len(stage))]
    layer_contrib = torch.cat([
        c[0].sum(1).sum(1) for c in contribution]).detach()
    score.append(layer_contrib)
    rank.append(layer_contrib.argsort(descending=True))
score = torch.stack(score)
rank = torch.stack(rank)
print(score.shape, rank.shape)
catid = 0
vis_imgs = []
for i in range(10):
    idx = rank[catid, i]
    stage_ind = cumdims.searchsorted(idx + 1) - 1
    stage_idx = int(idx - cumdims[stage_ind])
    img = stage[stage_ind][:, stage_idx:stage_idx+1]
    vis_imgs.append((img, i, idx, score[catid, idx]))

maxsize = max([img[0].shape[3] for img in vis_imgs])

for img, i, idx, val in vis_imgs:
    img = img / img.max() # the featuremap is positive
    if img.shape[3] <= 256:
        img = F.interpolate(img, size=256)
    img = utils.heatmap_torch(img)
    vutils.save_image(img, f"{i:02d}_{idx:04d}_{val:.3f}.png")

print("=> Show random subsamples")
indice = list(cr[C])
cname = utils.CELEBA_CATEGORY[C]
print(f"=> Category {cname} size {len(indice)}")
count = 0
print(indice)
for k in [10, 40, 100, 200, len(indice)]:
    repeat_num = 1 if k == len(indice) else 5
    for i in range(repeat_num): # repeat
        sample = np.random.choice(indice, size=k, replace=False)
        s = []
        for idx in sample:
            stage_ind = cumdims.searchsorted(idx + 1) - 1
            stage_idx = int(idx - cumdims[stage_ind])
            img = stage[stage_ind][0:1, stage_idx:stage_idx+1]
            s.append(img * w[C, idx])
        size = max([a.shape[2] for a in s])
        img = sum([F.interpolate(a, size=size, mode="bilinear", align_corners=True)
            for a in s])
        img = utils.heatmap_torch(img / img.max())
vutils.save_image(img, f"{cname}_sample.png")
count += 1
"""