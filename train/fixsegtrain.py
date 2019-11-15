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

cfg = config.FixSegConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dict = torch.load(cfg.seg_net_path, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

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

if cfg.zero_noise:
	noise = [0] * 18
	for k in range(18):
		size = 4 * 2 ** (k // 2)
		noise[k] = torch.zeros(cfg.batch_size, 1, size, size).cuda()
	sg.set_noise(noise)

record = cfg.record
avgmseloss = 0
count = 0

for i in tqdm(range(cfg.n_iter + 1)):
	latent.normal_()

	with torch.no_grad():
		gen = sg(latent)
		gen = F.interpolate(gen, 512, mode="bilinear")
		label = faceparser(gen).argmax(1)
		if cfg.map_id:
			label = cfg.idmap(label.detach())

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

	if (i % 1000 == 0 and i > 0) or i == cfg.n_iter:
		print("=> Snapshot model %d" % i)
		torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)

	if i % 1000 == 0 or i == cfg.n_iter or cfg.debug:
		vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % i,
							nrow=2, normalize=True, range=(-1, 1))

		tarlabels = [utils.tensor2label(label[i:i+1], label.shape[1])
						for i in range(label.shape[0])]
		tarlabels = [l.unsqueeze(0) for l in tarlabels]
		tarviz = torch.cat([F.interpolate(m, 256).cpu() for m in tarlabels])
		genlabels = [utils.tensor2label(s[0].argmax(0), s.shape[1])
					for s in segs]
		genlabels = [l.unsqueeze(0) for l in genlabels]
		gen_img = (gen[0:1].clamp(-1, 1) + 1) / 2
		genviz = genlabels + [gen_img]
		genviz = torch.cat([F.interpolate(m, 256).cpu() for m in genviz])
		vutils.save_image(genviz, cfg.expr_dir + "/genlabel_viz_%05d.png" % i, nrow=3)
		vutils.save_image(tarviz, cfg.expr_dir + "/tarlabel_viz_%05d.png" % i, nrow=2)
		utils.write_log(cfg.expr_dir, record)
		utils.plot_dic(record, cfg.expr_dir + "/loss.png")

#os.system(f"python script/monitor.py --task log,seg --model {cfg.expr_dir} --step 8 --gpu {cfg.gpu}")
#os.system(f"python test.py --model {cfg.expr_dir} --gpu {cfg.gpu}")