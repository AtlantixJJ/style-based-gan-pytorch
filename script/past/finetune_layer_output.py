"""
Train the RGB output of each resolution (to reduce memory cost of training on 1024x1024).
Turn out to be not working.
"""
import sys
sys.path.insert(0, ".")
import os
import torch
from torchvision import utils as vutils
from tqdm import tqdm
from PIL import Image
import numpy as np
import model
import config
from utils import *
from loss import *

cfg = config.BaseConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')

sg = model.tf.StyledGenerator()
missed = sg.load_state_dict(state_dicts, strict=False)
#sg.g_synthesis.torgb.weight.data.copy_(state_dicts['g_synthesis.torgb.weight'])
#sg.g_synthesis.torgb.bias.data.copy_(state_dicts['g_synthesis.torgb.bias'])
#sg.g_synthesis.torgbs[-1].weight.data.copy_(state_dicts['g_synthesis.torgb.weight'])
#sg.g_synthesis.torgbs[-1].bias.data.copy_(state_dicts['g_synthesis.torgb.bias'])
print(missed)
sg.train()
sg = sg.cuda()

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.g_synthesis.torgbs.parameters(), lr=1e-3)
mse = torch.nn.MSELoss()
mse = mse.cuda()

latent = torch.randn(cfg.batch_size, 512).cuda()

count = 0
record = {'loss': []}
for i in tqdm(range(10001)):
    latent.normal_()

    images = sg.all_layer_forward(latent)
    target = images[-1].detach()
    images = images[:-1]
    mselosses = []
    for img in images:
        mselosses.append(mse(
            F.interpolate(img, target.shape[2:], mode="bilinear", align_corners=True),
            target))
    mseloss = sum(mselosses) / len(mselosses)

    loss = mseloss
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    record['loss'].append(torch2numpy(loss))
    if i % 20 == 0:
        write_log(cfg.expr_dir, record)
        plot_dic(record, "loss", f"{cfg.expr_dir}/loss.png")

    if i % 1000 == 0:
        images = images + [target]
        image_list = [F.interpolate(img[0:1], (256, 256), mode="bilinear", align_corners=True) for img in images]
        image_list = (torch.cat(image_list).clamp(-1, 1) + 1) / 2
        vutils.save_image(image_list, f"{cfg.expr_dir}/img_{i // 200}.png", nrow=4)

torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)