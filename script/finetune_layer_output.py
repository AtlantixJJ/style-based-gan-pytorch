import sys
sys.path.insert(0, ".")
import os
import torch
from torchvision import utils as vutils
from tqdm import tqdm
from PIL import Image
import numpy as np
from model.seg import StyledGenerator
import config
from utils import *
from loss import *

STEP = 8
ALPHA = 1

cfg = config.BaseConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')

sg = StyledGenerator(512)
sg.load_state_dict(state_dicts['generator'])
sg.train()
sg = sg.cuda()

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.generator.to_rgb.parameters(), lr=cfg.lr, betas=(0.9, 0.9))
mse = torch.nn.MSELoss()
mse = mse.cuda()

latent = torch.randn(cfg.batch_size, 512).cuda()
noise = [0] * (STEP + 1)
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noise[k] = torch.randn(cfg.batch_size, 1, size, size).cuda()

count = 0
record = {'loss': []}
for i in tqdm(range(cfg.n_iter)):
    latent.normal_()
    for k in range(STEP + 1):
        noise[k].normal_()

    images = sg.all_level_forward(latent, step=STEP, alpha=ALPHA, noise=noise)
    target = images[-1].detach()
    images = images[:-1]
    mselosses = []
    for img in images:
        mselosses.append(mse(
            F.interpolate(img, target.shape[2:], mode="bilinear"),
            target))
    mseloss = sum(mselosses) / len(mselosses)

    loss = mseloss
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    record['loss'].append(torch2numpy(loss))
    write_log(cfg.expr_dir, record)

torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)