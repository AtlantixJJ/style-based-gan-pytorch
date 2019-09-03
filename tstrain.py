import os
import torch
import config
from utils import *
from model import StyledGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")


STEP = 7
ALPHA = 1


def set_lerp_val(progression, lerp_val):
    for p in progression:
        p.lerp_val = lerp_val


def get_generator_lr(g, lr1, lr2, stage=0):
    dic = []
    for blk in g.progression:
        if stage > 0:
            dic.append({"params": blk.conv1.parameters(), "lr": lr1})
            dic.append({"params": blk.conv2.parameters(), "lr": lr1})
        if blk.att > 0:
            dic.append({"params": blk.atthead1.parameters(), "lr": lr2})
            dic.append({"params": blk.atthead2.parameters(), "lr": lr2})
    if stage > 0:
        dic.append({"params": g.to_rgb.parameters(), "lr": lr1})
    return dic


def get_mask(styledblocks):
    masks = []
    for blk in styledblocks:
        if blk.att > 0:
            masks.extend(blk.mask1)
            masks.extend(blk.mask2)
    return masks


def maskarealoss(g, coef=1):
    masks = []
    for blk in g.generator.progression:
        if blk.att > 0:
            masks.extend(blk.mask1)
            masks.extend(blk.mask2)
    return coef * sum([(m.mean() - 0.5) ** 2 for m in masks])


def maskdivloss(g, coef=1):
    mask_divergence = 0
    count = 0
    for blk in sg.generator.progression:
        if blk.att > 0:
            for i in range(cfg.att):
                for j in range(i+1, cfg.att):
                    mask_divergence += (blk.mask1[i] * blk.mask1[j]).mean()
                    mask_divergence += (blk.mask2[i] * blk.mask2[j]).mean()
                    count += 2
    return mask_divergence / count


cfg = config.TSConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

lrschedule = PLComposite(0, 0.1)
lrschedule.add(cfg.stage1_step, 1)
lrschedule.add(cfg.stage1_step + 10, 0.1)
lrschedule.add(cfg.stage1_step * 2, 0.9)
lrschedule.add(cfg.stage2_step, 1)

tg = StyledGenerator(512).to(cfg.device1)
state_dict = torch.load(cfg.load_path)
tg.load_state_dict(state_dict)
tg.eval()
sg = StyledGenerator(512, att=cfg.att).to(cfg.device2)
sg.load_state_dict(state_dict)
sg.train()

g_optim1 = torch.optim.Adam(get_generator_lr(
    sg.generator, cfg.stage1_lr * 0.1, cfg.stage1_lr, 0), betas=(0.9, 0.9))
g_optim2 = torch.optim.Adam(get_generator_lr(
    sg.generator, cfg.stage2_lr, cfg.stage2_lr, 1), betas=(0.9, 0.9))
g_optim2.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.stage2_lr * 0.01})
g_optim3 = torch.optim.Adam(get_generator_lr(
    sg.generator, cfg.lr, cfg.lr, 1), betas=(0.9, 0.9))
g_optim3.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.lr * 0.01})
crit = torch.nn.MSELoss().cuda()

record = cfg.record
noise = [0] * (STEP + 1)
avgmseloss = 0
count = 0
for i in tqdm(range(cfg.n_iter + 1)):
    latent = torch.randn(cfg.batch_size, 512)
    for k in range(STEP + 1):
        size = 4 * 2 ** k
        noise[k] = torch.randn(cfg.batch_size, 1, size, size)

    with torch.no_grad():
        target_image = tg(latent.to(cfg.device1), noise=[n.to(
            cfg.device1) for n in noise], step=STEP, alpha=ALPHA)
        target_image = target_image.detach().to(cfg.device2)

    # if avgloss < 0.1:
    #    count += 1
    if i <= cfg.stage1_step:
        lerp_val = lrschedule(i)
        g_optim = g_optim1
    elif i <= cfg.stage2_step:
        lerp_val = lrschedule(i)
        g_optim = g_optim2
    else:
        lerp_val = 1
        g_optim = g_optim3
    set_lerp_val(sg.generator.progression, lerp_val)

    gen = sg(latent.to(cfg.device2), noise=[
             n.to(cfg.device2) for n in noise], step=STEP, alpha=ALPHA)
    mseloss = crit(gen, target_image)

    if cfg.ma > 0:
        mask_avgarea = maskarealoss(sg, cfg.ma)
    else:
        mask_avgarea = -1

    if cfg.md > 0:
        mask_divergence = maskdivloss(sg, cfg.md)
    else:
        mask_divergence = -1

    loss = mseloss + mask_avgarea + mask_divergence
    loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    mse_data = torch2numpy(mseloss)
    avgmseloss = avgmseloss * 0.9 + mse_data * 0.1
    record['loss'].append(torch2numpy(loss))
    record['lerp_val'].append(lerp_val)
    record['mseloss'].append(mse_data)
    if mask_avgarea > -1:
        record['mask_area'].append(torch2numpy(mask_avgarea))
    if mask_divergence > -1:
        record['mask_div'].append(torch2numpy(mask_divergence))

    if i % 1000 == 0 and i > 0:
        print("=> Snapshot model")
        torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)

    if i % 100 == 0:
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        vutils.save_image(target_image[:4], cfg.expr_dir + '/target_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        write_log(cfg.expr_dir, record)