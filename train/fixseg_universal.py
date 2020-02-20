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
import utils, evaluate, loss, model

from lib.netdissect.segviz import segment_visualization
from lib.netdissect.segmenter import UnifiedParsingSegmenter
from lib.netdissect import proggan
from lib.netdissect.zdataset import standard_z_sample, z_dataset_for_model

cfg = config.FixSegConfig()
cfg.parse()
s = str(cfg)
cfg.setup()
print(s)
        
latent = torch.randn(1, 512).to(cfg.device)
generator = model.load_model_from_pth_file(cfg.model_name, cfg.load_path).to(cfg.device)
generator.eval()
image, stage = generator.get_stage(latent, True)
dims = [s.shape[1] for s in stage]

segmenter = UnifiedParsingSegmenter()
labels, cats = segmenter.get_label_and_category_names()
with open(cfg.expr_dir + "/config.txt", "w") as f:
    f.write(s)
    for i, (label, cat) in enumerate(labels):
        f.write('%s %s\n' % (label, cat))


category_groups = utils.get_group(labels)
category_groups_label = utils.get_group(labels, False)
n_class = category_groups[-1][1]

seg = segmenter.segment_batch(image)
linear_model = model.linear.LinearSemanticExtractor(
    n_class, dims, None, category_groups).to(cfg.device)
output = linear_model(stage)

record = {"segloss": []}

for ind in tqdm(range(cfg.n_iter)):
    ind += 1
    latent.normal_()

    with torch.no_grad(): # fix main network
        gen, stage = generator.get_stage(latent, detach=True)
        label = segmenter.segment_batch(gen)

    multi_segs = linear_model(stage)
    segloss = 0
    for i, segs in enumerate(multi_segs):
        if label[:, i, :, :].max() <= 0:
            continue
        cg = category_groups_label[i]
        l = label[:, i, :, :] - cg[0]
        l[l<0] = 0
        segloss = segloss + loss.segloss(segs, l)

    segloss.backward()
    linear_model.optim.step()
    linear_model.optim.zero_grad()

    record['segloss'].append(utils.torch2numpy(segloss))
