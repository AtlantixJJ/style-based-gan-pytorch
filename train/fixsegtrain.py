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

STEP = 8 # 1024px resolution
ALPHA = 1

cfg = config.SConfig()
cfg.parse()
cfg.print_info()
cfg.setup()
# GPU1: main network + auxiliary network GPU2: extractor
state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(cfg.device)
faceparser.eval()
del state_dict

state_dicts = torch.load(cfg.load_path, map_location='cpu')
sg = StyledGenerator(512, semantic=cfg.semantic_config)
sg.load_state_dict(state_dicts['generator'])
sg.train()
sg = sg.to(cfg.device)
# the extractor module in on another GPU
#for i, blk in enumerate(sg.generator.progression):
#	if blk.n_class > 0:
#		blk.extractor = blk.extractor.to(cfg.device2)
del state_dicts

# new parameter adaption stage
g_optim = torch.optim.Adam(get_generator_extractor_lr(
    sg.generator, cfg.lr), betas=(0.9, 0.9))
logsoftmax = torch.nn.CrossEntropyLoss()
logsoftmax = logsoftmax.to(cfg.device2)

latent = torch.randn(cfg.batch_size, 512).to(cfg.device)
noise = [0] * (STEP + 1)
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noise[k] = torch.randn(cfg.batch_size, 1, size, size).to(cfg.device)

record = {"loss":[], "segloss":[]}
count = 0

for i in tqdm(range(cfg.n_iter + 1)):
	latent.normal_()
	for k in range(STEP + 1):
		noise[k].normal_()

	with torch.no_grad():
		gen = sg(latent,
			noise=noise,
			step=STEP,
			alpha=ALPHA)
		gen = F.interpolate(gen, cfg.imsize, mode="bilinear")
		label = faceparser(gen).argmax(1).detach()

	segs = []
	for j in range(STEP + 1):
		if sg.generator.progression[j].n_class <= 0:
			continue
		segmentation = get_segmentation(sg.generator.progression, j)[0]
		segs.append(segmentation)
		segloss = logsoftmax(F.interpolate(
			segmentation,
			label.shape[2:],
			mode="bilinear"), label)
		loss = segloss
		with torch.autograd.detect_anomaly():
			loss.backward()
		g_optim.step()
		g_optim.zero_grad()

	record['loss'].append(torch2numpy(loss))
	record['segloss'].append(torch2numpy(segloss))

	if cfg.debug:
		print(record.keys())
		l = []
		for k in record.keys():
			l.append(record[k][-1])
		print(l)

	if i % 100 == 0 and i > 0:
		print("=> Snapshot model %d" % i)
		torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)

	if i % 100 == 0 or cfg.debug:
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

os.system("python script/monitor.py --task log,seg --model %s --step 8" % cfg.expr_dir)