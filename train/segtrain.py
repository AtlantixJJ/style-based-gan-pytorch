import sys
sys.path.insert(0, ".")
import os
import torch
import config
from utils import *
from loss import *
from model import StyledGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

STEP = 8
ALPHA = 1

cfg = config.TSConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')

sg = StyledGenerator(512, att=cfg.att, att_mtd=cfg.att_mtd)
sg.load_state_dict(state_dicts['generator'])
sg.train()
sg = tg.to(cfg.device2)

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.generator.parameters(), betas=(0.9, 0.9))
g_optim.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.stage1_lr * 0.01})
logsoftmax = torch.nn.NLLLoss2d()
mse = torch.nn.MSELoss()

for k in range(STEP + 1):
    size = 4 * 2 ** k
    noise[k] = torch.randn(cfg.batch_size, 1, size, size).cuda()

record = cfg.record
noise = [0] * (STEP + 1)
avgmseloss = 0
count = 0

for i, (latent, image, label) in tqdm(enumerate(cfg.dl)):
    for k in range(STEP + 1):
        noise[k].normal_()

    if i <= cfg.stage2_step:
        lerp_val = lrschedule(i)
        g_optim = g_optim1
    else:
        lerp_val = 1
        g_optim = g_optim2
    set_lerp_val(sg.generator.progression, lerp_val)

    gen = sg(latent.cuda(), noise=noise, step=STEP, alpha=ALPHA)
    mseloss = mse(gen, image)
    segs = get_segmentation(sg.generator.progression)
    seglosses = []
    for s in segs:
        seglosses.append(crit(
            F.interpolate(gen, label.shape[2:], mode="bilinear"),
            label))
    segloss = cfg.seg_coef * sum(seglosses) / len(seglosses)

    loss = mseloss + segloss
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    avgmseloss = avgmseloss * 0.9 + mse_data * 0.1
    record['loss'].append(torch2numpy(loss))
    record['lerp_val'].append(lerp_val)
    record['mseloss'].append(torch2numpy(mseloss))
    record['segloss'].append(torch2numpy(segloss))

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

    if i % 1000 == 0 and i > 0:
        print("=> Snapshot model %d" % i)
        torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        vutils.save_image(target_image[:4], cfg.expr_dir + '/target_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        write_log(cfg.expr_dir, record)
