"""
Teacher student training of SRGAN.
"""
import sys
sys.path.insert(0, ".")
import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing.utils import tensor2label
from lib.face_parsing import unet
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

state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(cfg.device2)
faceparser.eval()
del state_dict

state_dicts = torch.load(cfg.load_path, map_location='cpu')
tg = StyledGenerator(512)
tg.load_state_dict(state_dicts['generator'])
tg.eval()
tg = tg.to(cfg.device2)
sg = StyledGenerator(512, semantic=cfg.semantic_config)
sg.load_state_dict(state_dicts['generator'])
sg.train()
sg = sg.to(cfg.device1)
del state_dicts

# new parameter adaption stage
g_optim1 = torch.optim.Adam(get_generator_extractor_lr(
    sg.generator, cfg.lr), betas=(0.9, 0.9)) # 1e-3
g_optim2 = torch.optim.Adam(get_generator_extractor_lr(
    sg.generator, cfg.lr), betas=(0.9, 0.9))
params = get_generator_blockconv_lr(sg.generator, cfg.lr * 0.1) # 1e-4
for p in params:
	g_optim2.add_param_group(d)
logsoftmax = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()
logsoftmax = logsoftmax.to(cfg.device1)
mse = mse.cuda(cfg.device1)

latent1 = torch.randn(cfg.batch_size, 512).to(cfg.device1)
latent2 = latent1.clone().to(cfg.device2)
noise1 = [0] * (STEP + 1)
noise2 = []
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noise1[k] = torch.randn(cfg.batch_size, 1, size, size).to(cfg.device1)
for k in range(STEP + 1):
    noise2.append(noise1[k].clone().to(cfg.device2))

record = cfg.record
avgmseloss = 0
count = 0

for i in tqdm(range(cfg.n_iter)):
	latent1.normal_()
	latent2.copy_(latent1, True) # asynchronous
	for k in range(STEP + 1):
		noise1[k].normal_()
		noise2[k].copy_(noise2[k], True) # asynchronous

	gen = sg(latent1, step=STEP, alpha=ALPHA, noise=noise1)
	with torch.no_grad():
		image = tg(latent2, noise=noise2, step=STEP, alpha=ALPHA)
		image = F.interpolate(image, cfg.imsize, mode="bilinear")
		label = faceparser(image).argmax(1)
		image = image.detach().cpu().to(cfg.device1)
		label = label.detach().cpu().to(cfg.device1)

	mseloss = cfg.mse_coef * mse(F.interpolate(gen, cfg.imsize, mode="bilinear"), image)
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
	
	if i < 1000:
		g_optim = g_optim1
	else:
		g_optim = g_optim2

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

		tarlabels = [torch.from_numpy(tensor2label(
						label[i:i+1],
						label.shape[1]))
						for i in range(label.shape[0])]
		tarlabels = [l.float().unsqueeze(0) for l in tarlabels]
		tarviz = torch.cat([F.interpolate(m, 256).cpu() for m in tarlabels])
		genlabels = [torch.from_numpy(tensor2label(s[0], s.shape[1]))
					for s in segs]
		genlabels = [l.float().unsqueeze(0) for l in genlabels]
		genviz = genlabels + [(gen[0:1] + 1) / 2]
		genviz = torch.cat([F.interpolate(m, 256).cpu() for m in genviz])
		vutils.save_image(genviz, cfg.expr_dir + "/genlabel_viz_%05d.png" % i, nrow=3)
		vutils.save_image(tarviz, cfg.expr_dir + "/tarlabel_viz_%05d.png" % i, nrow=2)

		write_log(cfg.expr_dir, record)
