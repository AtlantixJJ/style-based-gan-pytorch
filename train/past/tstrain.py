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

STEP = 7
ALPHA = 1

cfg = config.TSConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

lrschedule = PLComposite(0, 0.1)
#lrschedule.add(cfg.stage1_step, 1)
#lrschedule.add(cfg.stage1_step + 10, 0.1)
#lrschedule.add(cfg.stage1_step * 2, 0.9)
lrschedule.add(cfg.stage2_step, 1)

state_dicts = torch.load(cfg.load_path, map_location='cpu')

disc = Discriminator()
disc.load_state_dict(state_dicts['discriminator'])
tg = StyledGenerator(512).to(cfg.device1)
tg.load_state_dict(state_dicts['generator'])
tg.eval()
sg = StyledGenerator(512, att=cfg.att, att_mtd=cfg.att_mtd)
sg.load_state_dict(state_dicts['generator'])
sg.train()

disc = disc.to(cfg.device1)
tg = tg.to(cfg.device1)
sg = tg.to(cfg.device2)

# new parameter adaption stage
g_optim1 = torch.optim.Adam(get_generator_lr(
    sg.generator, cfg.stage1_lr * 0.1, cfg.stage1_lr), betas=(0.9, 0.9))
g_optim1.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.stage1_lr * 0.001})
# normal learning rate stage
g_optim2 = torch.optim.Adam(get_generator_lr(
    sg.generator, cfg.stage2_lr, cfg.stage2_lr), betas=(0.9, 0.9))
g_optim2.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.stage2_lr * 0.01})
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
    if i <= cfg.stage2_step:
        lerp_val = lrschedule(i)
        g_optim = g_optim1
    else:
        lerp_val = 1
        g_optim = g_optim2
    set_lerp_val(sg.generator.progression, lerp_val)

    gen = sg(latent.to(cfg.device2), noise=[
             n.to(cfg.device2) for n in noise], step=STEP, alpha=ALPHA)
    mseloss = crit(gen, target_image)

    if cfg.ma > 0:
        mask_avgarea = maskarealoss(
            sg.generator.progression, target=1.0/cfg.att, coef=cfg.ma)
    else:
        mask_avgarea = 0

    if cfg.md > 0:
        mask_divergence = maskdivloss(sg.generator.progression, coef=cfg.md)
    else:
        mask_divergence = 0

    if cfg.mv > 0:
        mask_value = maskvalueloss(
            sg.generator.progression, coef=cfg.mv)
    else:
        mask_value = 0

    loss = mseloss + mask_avgarea + mask_divergence + mask_value
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    mse_data = torch2numpy(mseloss)
    avgmseloss = avgmseloss * 0.9 + mse_data * 0.1
    record['loss'].append(torch2numpy(loss))
    record['lerp_val'].append(lerp_val)
    record['mseloss'].append(mse_data)
    if mask_avgarea > 0:
        record['mask_area'].append(torch2numpy(mask_avgarea))
    if mask_divergence > 0:
        record['mask_div'].append(torch2numpy(mask_divergence))
    if mask_value > 0:
        record['mask_value'].append(torch2numpy(mask_value))

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
