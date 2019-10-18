import sys
sys.path.insert(0, ".")
import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing.utils import tensor2label
import config
from utils import *
from loss import *
from model.seg import StyledGenerator

STEP = 8
ALPHA = 1

cfg = config.SConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')

sg = StyledGenerator(512, semantic=cfg.semantic_config)
sg.load_state_dict(state_dicts['generator'])
sg.train()
sg = sg.cuda()

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.generator.parameters(), lr=cfg.lr, betas=(0.9, 0.9))
g_optim.add_param_group({
    'params': sg.style.parameters(),
    'lr': cfg.lr * 0.01})
logsoftmax = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()
logsoftmax = logsoftmax.cuda()
mse = mse.cuda()

#noise = [0] * (STEP + 1)
#for k in range(STEP + 1):
#    size = 4 * 2 ** k
#    noise[k] = torch.randn(cfg.batch_size, 1, size, size).cuda()

record = cfg.record
avgmseloss = 0
count = 0

for i, (latent, noise, image, label) in tqdm(enumerate(cfg.dl)):
    #for k in range(STEP + 1):
    #    noise[k].normal_()

    latent = latent.cuda()
    noise = [n.cuda() for n in noise]
    image = image.cuda()
    label = label.cuda()

    gen = sg(latent, step=STEP, alpha=ALPHA, noise=noise)
    mseloss = mse(F.interpolate(gen, image.shape[2:], mode="bilinear"), image)
    segs = get_segmentation(sg.generator.progression)
    seglosses = []
    for s in segs:
        seglosses.append(logsoftmax(
            F.interpolate(s, label.shape[2:], mode="bilinear"),
            label))
    segloss = cfg.seg_coef * sum(seglosses) / len(seglosses)

    loss = mseloss + segloss
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    record['loss'].append(torch2numpy(loss))
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

    if i % 1000 == 0 or cfg.debug:
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        vutils.save_image(image[:4], cfg.expr_dir + '/target_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        labels = [torch.from_numpy(tensor2label(s[0], s.shape[1]))
                    for s in segs]
        labels = [l.float().unsqueeze(0) for l in labels]
        res = labels + [(gen[0:1] + 1) / 2]
        res = torch.cat([F.interpolate(m, 256).cpu() for m in res])
        vutils.save_image(res, cfg.expr_dir + "/label_viz_%05d.png" % i, nrow=4)
        write_log(cfg.expr_dir, record)
