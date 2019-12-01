"""
Teacher student training of SRGAN.
"""
import sys
sys.path.insert(0, ".")
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing import unet
import config
import utils
from loss import *
import model

cfg = config.SegConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')
sg = getattr(model, cfg.arch).StyledGenerator(semantic=cfg.semantic_config)
sg.load_state_dict(state_dicts, strict=False)
sg.train()
sg = sg.cuda()
sg.freeze_g_mapping() # fix main trunk
sg.freeze_g_synthesis() # fix main trunk
del state_dicts

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.semantic_branch.parameters(),
	lr=cfg.lr,
	betas=(0.9, 0.9))
logsoftmax = torch.nn.CrossEntropyLoss()
logsoftmax = logsoftmax.cuda()

latent = torch.randn(cfg.batch_size, 512).cuda()

record = cfg.record
avgmseloss = 0
count = 0
noise = None

def infinite_dataloader(dl, total):
	i = 0
	while True:
		for sample in dl:
			i += 1
			if i == total:
				return
			yield sample
		dl.reset()

pbar = tqdm(infinite_dataloader(cfg.ds, cfg.n_iter + 1), total=cfg.n_iter + 1)
for ind, sample in enumerate(pbar):
	latent = torch.from_numpy(sample[0]).float().cuda()
	label = torch.from_numpy(sample[3]).long().cuda().unsqueeze(0)
	#if noise is None:
	#	noise = [torch.from_numpy(noise).float().cuda() for noise in sample[1]]
	#else:
	#	for i in range(len(noise)):
	#		noise[i] = torch.from_numpy(sample[1][i]).float().cuda()
	#sg.set_noise(noise)
	gen, seg = sg.predict(latent)
	gen = (gen + 1) / 2
	gen = gen.detach().cpu()

	if cfg.map_id:
		label = utils.idmap(label.detach())

	segs = sg.extract_segmentation()
	coefs = [1. for s in segs]
	seglosses = []
	for c, s in zip(coefs, segs):
		# s: (batch, 19, h, w); label: (batch, h, w)
		if s.shape[2] < label.shape[2]:
			label_ = F.interpolate(
				label.unsqueeze(1).float(), s.shape[2:],
				mode="bilinear")
			label_ = label_.squeeze(1).long()
			l = logsoftmax(s, label_)
		elif s.shape[2] > label.shape[2]:
			l = logsoftmax(
				F.interpolate(s, label.shape[2:], mode="bilinear"),
				label)
		else:
			l = logsoftmax(s, label)
		seglosses.append(c * l)
	segloss = cfg.seg_coef * sum(seglosses) / len(seglosses)

	loss = segloss
	with torch.autograd.detect_anomaly():
		loss.backward()

	g_optim.step()
	g_optim.zero_grad()

	record['loss'].append(utils.torch2numpy(loss))
	record['segloss'].append(utils.torch2numpy(segloss))

	if cfg.debug:
		print(record.keys())
		l = []
		for k in record.keys():
			l.append(record[k][-1])
		print(l)

	if (ind % 500 == 0 and ind > 0) or ind == cfg.n_iter:
		print("=> Snapshot model %d" % ind)
		torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % ind)

	if ind % 500 == 0 or ind == cfg.n_iter or cfg.debug:
		tarlabel = utils.tensor2label(label[0], label.shape[1]).unsqueeze(0)
		genlabels = [utils.tensor2label(s[0], s.shape[1]).unsqueeze(0)
					for s in segs]
		viz = genlabels + [gen, tarlabel]
		viz = torch.cat([F.interpolate(m, 256).cpu() for m in viz])
		vutils.save_image(viz, cfg.expr_dir + "/viz_%05d.png" % ind, nrow=2)
		utils.write_log(cfg.expr_dir, record)
		utils.plot_dic(record, cfg.expr_dir + "/loss.png")