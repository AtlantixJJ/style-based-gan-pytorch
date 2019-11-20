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
import config
import utils
from lib.face_parsing import unet
from loss import *
import model

cfg = config.TSSegConfig()
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
tg = model.tf.StyledGenerator()
tg.load_state_dict(state_dicts)
tg.eval()
tg = tg.to(cfg.device2)
sg = getattr(model, cfg.arch).StyledGenerator(semantic=cfg.semantic_config)
sg.load_state_dict(state_dicts, strict=False)
sg.train()
sg = sg.to(cfg.device1)
sg.freeze_g_mapping() # do not change style branch
sg.freeze_g_synthesis()
del state_dicts

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.semantic_branch.parameters(),
	lr=cfg.lr,
	betas=(0.9, 0.9)) # 1e-3
g_optim.add_param_group({
    'params': sg.g_synthesis.parameters(),
    'lr': cfg.lr * 0.1})
logsoftmax = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()
logsoftmax = logsoftmax.to(cfg.device1)
mse = mse.cuda(cfg.device1)

latent1 = torch.randn(cfg.batch_size, 512).to(cfg.device1)
latent2 = latent1.clone().to(cfg.device2)
noise1 = [0] * 18
noise2 = []
for k in range(18):
    size = 4 * 2 ** (k // 2)
    noise1[k] = torch.randn(cfg.batch_size, 1, size, size).to(cfg.device1)
for k in range(18):
    noise2.append(noise1[k].clone().to(cfg.device2))

record = cfg.record
avgmseloss = 0
count = 0

for i in tqdm(range(cfg.n_iter + 1)):
	if i == 1001:
		sg.freeze_g_synthesis(train=True)

	latent1.normal_()
	latent2.copy_(latent1, True) # asynchronous
	for k in range(18):
		noise1[k].normal_()
		noise2[k].copy_(noise2[k], True) # asynchronous
	sg.set_noise(noise1)
	tg.set_noise(noise2)

	gen = sg(latent1)
	with torch.no_grad():
		image = tg(latent2)
		image = F.interpolate(image, cfg.imsize, mode="bilinear")
		label = faceparser(image).argmax(1)
		image = image.detach().cpu().to(cfg.device1)
		label = label.detach().cpu().to(cfg.device1)

	mseloss = cfg.mse_coef * mse(F.interpolate(gen, cfg.imsize, mode="bilinear"), image)
	segs = sg.extract_segmentation()
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

	record['loss'].append(utils.torch2numpy(loss))
	record['mseloss'].append(utils.torch2numpy(mseloss))
	record['segloss'].append(utils.torch2numpy(segloss))

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

		tarlabels = [utils.tensor2label(label[i:i+1], label.shape[1]).unsqueeze(0)
						for i in range(label.shape[0])]
		tarviz = torch.cat([F.interpolate(m, 256).cpu() for m in tarlabels])
		genlabels = [utils.tensor2label(s[0], s.shape[1]).unsqueeze(0)
					for s in segs]
		gen_img = (gen[0:1].clamp(-1, 1) + 1) / 2
		genviz = genlabels + [gen_img]
		genviz = torch.cat([F.interpolate(m, 256).cpu() for m in genviz])
		vutils.save_image(genviz, cfg.expr_dir + "/genlabel_viz_%05d.png" % i, nrow=3)
		vutils.save_image(tarviz, cfg.expr_dir + "/tarlabel_viz_%05d.png" % i, nrow=2)
		utils.write_log(cfg.expr_dir, record)
		utils.plot_dic(record, cfg.expr_dir + "/loss.png")

#os.system(f"python script/monitor.py --task log,seg --model {cfg.expr_dir} --step 8 --gpu {cfg.gpu}")
#os.system(f"python test.py --model {cfg.expr_dir} --gpu {cfg.gpu}")