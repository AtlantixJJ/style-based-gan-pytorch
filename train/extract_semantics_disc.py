import torch, numpy, os, argparse, sys, shutil
sys.path.insert(0, ".")
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing import unet
import config
import utils, evaluate, loss, segmenter, model

from model.semantic_extractor import get_semantic_extractor
from lib.netdissect.segviz import segment_visualization, segment_visualization_single
from lib.netdissect import proggan
from lib.netdissect.zdataset import standard_z_sample, z_dataset_for_model

cfg = config.DDSEConfig()
cfg.parse()
cfg.setup()

n_class = cfg.n_class
print(str(cfg))
with open(cfg.expr_dir + "/config.txt", "w") as f:
    f.write(str(cfg))
disc = model.tf.Discriminator(resolution=cfg.imsize)
if cfg.load:
    disc.load_state_dict(torch.load(cfg.load_path))
disc.to(cfg.device).eval()

x = torch.randn(1, 3, cfg.imsize, cfg.imsize, device=cfg.device)
with torch.no_grad():
    image, stage = disc.get_stage(x)
stage = [s for i, s in enumerate(stage) if i in cfg.layers]
dims = [s.shape[1] for s in stage]
print(dims)
sep_model = get_semantic_extractor(cfg.semantic_extractor)(
    n_class=n_class,
    dims=dims).to(cfg.device)

record = cfg.record
metric = evaluate.DetectionMetric(n_class=n_class)

colorizer = utils.Colorize(n_class)
if cfg.task != "celebahq" and cfg.task != "ffhq":
    colorizer = lambda x: segment_visualization_single(x, 256)

#import objgraph
#t1 = t2 = 0

for ind, sample in enumerate(tqdm(cfg.dl)): #for ind in range(cfg.n_iter):
    ind += 1
    if ind > cfg.n_iter:
        break
    image, label = sample #torch.rand(1, 3, 1024, 1024), torch.ones(1, 1, 1024, 1024, 1).long() #
    image = image.to(cfg.device)
    label = label[:, 0, :, :, 0].to(cfg.device)

    with torch.no_grad(): # fix main network
        d, stage = disc.get_stage(image)
        stage = [s for i, s in enumerate(stage) if i in cfg.layers]
    segs = sep_model(stage, last_only=cfg.last_only)

    if segs[-1].size(3) < label.shape[2]:
        segs[-1] = F.interpolate(
            segs[-1], size=label.shape[2], mode="bilinear")

    segloss = loss.segloss(segs, label)

    regloss = 0
    if cfg.l1_reg > 0:
        regloss = regloss + cfg.l1_reg * loss.l1(sep_model)
    if cfg.norm_reg > 0:
        regloss = regloss + cfg.norm_reg * loss.l1norm(sep_model.semantic_extractor)
    total_loss = segloss + regloss
    total_loss.backward()
    sep_model.optim.step()
    sep_model.optim.zero_grad()

    # collect training statistic
    est_label = segs[-1].argmax(1)
    for j in range(est_label.shape[0]):
        metric(
            utils.torch2numpy(est_label[j]),
            utils.torch2numpy(label[j]))
    record['loss'].append(utils.torch2numpy(total_loss))
    record['segloss'].append(utils.torch2numpy(segloss))
    record['regloss'].append(utils.torch2numpy(regloss))

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        # visualize training
        res = []
        size = label.shape[1:]
        image = F.interpolate(image, size=size, mode="bilinear")
        for i in range(label.shape[0]): # label (N, M, H, W)
            image = (utils.torch2numpy(image[i]) + 1) * 127.5
            res.append(image.transpose(1, 2, 0))
            l = utils.torch2numpy(label[i])
            label_viz = colorizer(l)
            seg = utils.torch2numpy(segs[-1][i]).argmax(0)
            seg_viz = colorizer(seg)
            res.extend([seg_viz, label_viz])
        res = torch.from_numpy(np.stack(res)).permute(0, 3, 1, 2).float() / 255.
        vutils.save_image(res, f"{cfg.expr_dir}/{ind:05d}.png",
            nrow=1 + 2 * label.shape[1])

        # write log
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", f"{cfg.expr_dir}/loss.png")

        # show metric
        metric.aggregate(ind + 1 - cfg.disp_iter)
        print(metric)
    
    """
    if ind == 1:
        objgraph.show_growth()
    if ind == 20:
        objgraph.show_growth()

    if ind == 1:
        t1 = objgraph.by_type('dict')
    if ind == 20:
        t2 = objgraph.by_type('dict')
        count = 0
        for t in t2:
            flag = False
            for x in t1:
                if x is t:
                    flag = True
                    break
            if flag:
                continue
            objgraph.show_backrefs(t,
                max_depth=5, filename=f'obj{count}.dot')
            count += 1
        break
    """

torch.save(sep_model.state_dict(), f"{cfg.expr_dir}/{cfg.model_name}_{cfg.semantic_extractor}_extractor.model")
np.save(f"{cfg.expr_dir}/training_evaluation.npy", [metric.result])
