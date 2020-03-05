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

cfg = config.SemanticExtractorConfig()
cfg.parse()
cfg.setup()

external_model = segmenter.get_segmenter(cfg.task, cfg.seg_net_path)
labels, cats = external_model.get_label_and_category_names()
category_groups = utils.get_group(labels)
category_groups_label = utils.get_group(labels, False)
n_class = category_groups[-1][1]
print(str(cfg))
with open(cfg.expr_dir + "/config.txt", "w") as f:
    f.write(str(cfg))
    f.write("\n\n")
    for i, (label, cat) in enumerate(labels):
        f.write('%s %s\n' % (label, cat))
    f.write("\n%s\n" % str(category_groups))
    f.write("\n%s\n" % str(category_groups_label))

latent = torch.randn(cfg.batch_size, 512).to(cfg.device)
generator = model.load_model_from_pth_file(cfg.model_name, cfg.load_path)
generator.to(cfg.device).eval()

image, stage = generator.get_stage(latent, True)
dims = [s.shape[1] for s in stage]
sep_model = get_semantic_extractor(cfg.semantic_extractor)(
    n_class=n_class,
    dims=dims,
    mapid=None,
    category_groups=category_groups).to(cfg.device)

is_resize = cfg.semantic_extractor != "spherical"
record = cfg.record
metrics = [evaluate.DetectionMetric(n_class=cg[1]-cg[0]) for i, cg in enumerate(category_groups)]

colorizer = utils.Colorize(16)
if cfg.task != "celebahq" and cfg.task != "ffhq":
    colorizer = lambda x: segment_visualization_single(x, 256)

for ind in tqdm(range(cfg.n_iter)):
    ind += 1
    latent.normal_()

    with torch.no_grad(): # fix main network
        gen, stage = generator.get_stage(latent, detach=True)
        gen = gen.clamp(-1, 1)
        label = 0
        if not is_resize:
            label = external_model.segment_batch(gen, resize=False)
        else:
            label = external_model.segment_batch(gen)

    multi_segs = sep_model(stage, last_only=cfg.last_only)
    if len(category_groups_label) == 1:
        multi_segs = [multi_segs]
        label = label.unsqueeze(1)
    segloss = 0
    for i, segs in enumerate(multi_segs):
        if label[:, i].max() <= 0:
            continue
        cg = category_groups_label[i]
        l = label[:, i, :, :] - cg[0]
        l[l<0] = 0
        if "KL" == cfg.loss_type:
            logits = external_model.seg
            l = F.softmax(logits, dim=1)
            segloss = segloss + loss.kl_div(segs, l)
        else:
            segloss = segloss + loss.segloss(segs, l)

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
    for i, segs in enumerate(multi_segs):
        est_label = segs[-1].argmax(1)
        for j in range(est_label.shape[0]):
            metrics[i](utils.torch2numpy(est_label[j]), utils.torch2numpy(l[j]))
    record['loss'].append(utils.torch2numpy(total_loss))
    record['segloss'].append(utils.torch2numpy(segloss))
    record['regloss'].append(utils.torch2numpy(regloss))

    if (ind + 1) % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        # visualize training
        res = []
        size = label.shape[2:]
        gen = F.interpolate(gen, size=size, mode="bilinear")
        for i in range(label.shape[0]): # label (N, M, H, W)
            image = (utils.torch2numpy(gen[i]) + 1) * 127.5
            res.append(image.transpose(1, 2, 0))
            for j in range(label.shape[1]):
                l = utils.torch2numpy(label[i, j])
                label_viz = colorizer(l)
                seg = utils.torch2numpy(multi_segs[j][-1][i]).argmax(0)
                seg[seg > 0] += category_groups_label[j][0]
                seg_viz = colorizer(seg)
                res.extend([seg_viz, label_viz])
        res = torch.from_numpy(np.stack(res)).permute(0, 3, 1, 2).float() / 255.
        vutils.save_image(res, f"{cfg.expr_dir}/{ind+1:05d}.png", nrow=1 + 2 * label.shape[1])

        # write log
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", f"{cfg.expr_dir}/loss.png")

        # show metric
        for i in range(len(category_groups)):
            metrics[i].aggregate(ind + 1 - cfg.disp_iter)
            print(metrics[i])

torch.save(sep_model.state_dict(), f"{cfg.expr_dir}/{cfg.model_name}_{cfg.semantic_extractor}_extractor.model")
np.save(f"{cfg.expr_dir}/training_evaluation.npy", [metrics[i].result for i in range(len(metrics))])
