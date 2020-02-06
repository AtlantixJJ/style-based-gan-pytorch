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
import model

def l2loss(x):
	return (x ** 2).sum()

cfg = config.FixSegConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dict = torch.load(cfg.seg_net_path, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(cfg.device2)
faceparser.eval()
del state_dict

state_dicts = torch.load(cfg.load_path, map_location='cpu')
if cfg.arch == "simple":
	upsample = int(np.log2(cfg.imsize // 4))
	sg = model.simple.Generator(upsample=upsample, semantic=cfg.semantic_config)
	LATENT_SIZE = 128
else:
	sg = getattr(model, cfg.arch).StyledGenerator(semantic=cfg.semantic_config)
	LATENT_SIZE = 512
sg.load_state_dict(state_dicts, strict=False)
sg.train()
sg = sg.to(cfg.device1)
sg.semantic_branch = sg.semantic_branch.to(cfg.device2)
sg.freeze() # fix main trunk
del state_dicts

# new parameter adaption stage
g_optim = torch.optim.Adam(sg.semantic_branch.parameters(),
	lr=cfg.lr,
	betas=(0.9, 0.9))
logsoftmax = torch.nn.CrossEntropyLoss()
logsoftmax = logsoftmax.cuda()

latent = torch.randn(cfg.batch_size, LATENT_SIZE).cuda()

record = cfg.record
trace_weight = 0
trace_bias = 0
if cfg.trace_weight:
	conv_dim = sg.semantic_branch.weight.shape[1]
	bias_dim = sg.semantic_branch.bias.shape[0]
	trace_weight = np.zeros((cfg.n_iter, cfg.n_class, conv_dim), dtype="float16")
	trace_bias = np.zeros((cfg.n_iter, bias_dim, cfg.n_class), dtype="float16")

evaluator = utils.MaskCelebAEval()

for ind in tqdm(range(cfg.n_iter)):
	ind += 1
	latent.normal_()

	with torch.no_grad(): # fix main network
		gen = sg(latent, seg=False).clamp(-1, 1)
		gen = F.interpolate(gen, cfg.seg_net_imsize, mode="bilinear")
		label = faceparser(gen).argmax(1)
		if cfg.map_id:
			label = cfg.idmap(label.detach())

	segs = sg.extract_segmentation(sg.stage)
	coefs = [1. for s in segs]
	seglosses = []
	for c, s in zip(coefs, segs):
		s_ = F.interpolate(s, label.size(2), mode="bilinear")
		l = logsoftmax(s_, label)
		seglosses.append(c * l)
	segloss = cfg.seg_coef * sum(seglosses) / len(seglosses)

	regloss = 0
	if "l2class" in cfg.semantic_config:
		# the sum of a class is regularized
		regloss = cfg.l2_lambda * l2loss(sg.semantic_branch.weight.abs().sum(1))
		regloss+= cfg.l2_lambda * l2loss(sg.semantic_branch.bias.abs())
	if "l2" in cfg.semantic_config:
		regloss = cfg.l2_lambda * l2loss(sg.semantic_branch.weight)
		regloss+= cfg.l2_lambda * l2loss(sg.semantic_branch.bias)

	loss = segloss + regloss
	loss.backward()

	g_optim.step()
	g_optim.zero_grad()

	record['loss'].append(utils.torch2numpy(loss))
	record['segloss'].append(utils.torch2numpy(segloss))
	record['regloss'].append(utils.torch2numpy(regloss))

	gen_label = F.interpolate(segs[-1], label.size(2), mode="bilinear").argmax(1)
	gen_label_np = gen_label.cpu().numpy()
	label_np = label.cpu().numpy()
	for i in range(latent.shape[0]):
		scores = evaluator.compute_score(gen_label_np[i], label_np[i])
		evaluator.accumulate(scores)

	if cfg.trace_weight:
		trace_weight[ind - 1] = utils.torch2numpy(sg.semantic_branch.weight)[:, :, 0, 0]
		trace_bias[ind - 1] = utils.torch2numpy(sg.semantic_branch.bias)

	if cfg.debug:
		print(record.keys())
		l = []
		for k in record.keys():
			l.append(record[k][-1])
		print(l)

	if (ind % 1000 == 0 and ind > 0) or ind == cfg.n_iter:
		print("=> Snapshot model %d" % ind)
		evaluator.aggregate()
		evaluator.summarize()
		evaluator.save(cfg.expr_dir + "/training_evaluation.npy")
		torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % ind)
		if cfg.trace_weight:
			np.save(cfg.expr_dir + "/trace_weight.npy", trace_weight[:ind])
			np.save(cfg.expr_dir + "/trace_bias.npy", trace_bias[:ind])

	if ind % 1000 == 0 or ind == cfg.n_iter or cfg.debug:
		num = min(4, label.shape[0])
		res = []
		for i in range(num):
			gen_seg = segs[0][i].argmax(0)
			tarlabel = utils.tensor2label(label[i], cfg.n_class)
			genlabel = utils.tensor2label(gen_seg, cfg.n_class)
			image = (gen[i].clamp(-1, 1) + 1) / 2
			res.extend([image, genlabel, tarlabel])
		res = torch.cat([F.interpolate(m.float().unsqueeze(0), 256).cpu() for m in res])
		vutils.save_image(res, cfg.expr_dir + "/%05d.png" % ind, nrow=3)
		utils.write_log(cfg.expr_dir, record)
		utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")