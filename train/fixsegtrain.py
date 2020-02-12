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

def conv2vec(convs):
	vecs = [utils.torch2numpy(conv[0].weight)[:, :, 0, 0] for conv in convs]
	return np.concatenate(vecs, 1)

def positive_convs(convs):
	ortho_loss = 0
	count = 0
	for conv in convs:
		w = conv[0].weight[:, :, 0, 0] # (out_dim, in_dim)
		#ww = torch.matmul(w, w.permute(1, 0))
		#I = torch.eye(ww.shape[0], device=ww.device)
		#ortho_loss += F.mse_loss(ww, I)
		ortho_loss += F.relu(w).sum()
		count += 1
	return ortho_loss / count

def ortho_convs(convs):
	ortho_loss = 0
	count = 0
	for conv in convs:
		w = conv[0].weight[:, :, 0, 0] # (out_dim, in_dim)
		ww = torch.matmul(w, w.permute(1, 0))
		I = torch.eye(ww.shape[0], device=ww.device)
		ortho_loss += F.mse_loss(ww, I)
		#ortho_loss += F.relu(w).sum()
		count += 1
	return ortho_loss / count

record = cfg.record
trace_weight = 0
if cfg.trace_weight and "conv" in cfg.semantic_config:
	vec = conv2vec(sg.semantic_branch.children())
	trace_weight = np.zeros((cfg.n_iter, vec.shape[0], vec.shape[1]), dtype="float16")

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

	# The last one is usually summation of previous one
	if not cfg.train_last:
		segs = segs[:-1]

	# calculate segmentation loss
	seglosses = []
	for s in segs:
		layer_loss = 0
		# label is large : downsample label
		if s.size(2) < label.size(2): 
			l_ = label.unsqueeze(0).float()
			l_ = F.interpolate(l_, s.size(2), mode="nearest")
			layer_loss = logsoftmax(s, l_.long()[0])
		# label is small : downsample seg
		elif s.size(2) > label.size(2): 
			s_ = F.interpolate(s, label.size(2), mode="bilinear")
			layer_loss = logsoftmax(s_, label)
		seglosses.append(layer_loss)
	segloss = sum(seglosses) / len(seglosses)

	regloss = 0

	if cfg.ortho_reg > 0:
		ortho_loss = ortho_convs(sg.semantic_branch.children())
		regloss = regloss + cfg.ortho_reg * ortho_loss

	if cfg.positive_reg > 0:
		ortho_loss = positive_convs(sg.semantic_branch.children())
		regloss = regloss + cfg.ortho_reg * ortho_loss

	loss = segloss + regloss
	loss.backward()

	g_optim.step()
	g_optim.zero_grad()

	record['loss'].append(utils.torch2numpy(loss))
	record['segloss'].append(utils.torch2numpy(segloss))
	record['regloss'].append(utils.torch2numpy(regloss))

	# calculate training accuracy
	gen_label = F.interpolate(segs[-1], label.size(2), mode="bilinear").argmax(1)
	gen_label_np = gen_label.cpu().numpy()
	label_np = label.cpu().numpy()
	for i in range(latent.shape[0]):
		scores = evaluator.compute_score(gen_label_np[i], label_np[i])
		evaluator.accumulate(scores)

	if cfg.trace_weight:
		trace_weight[ind - 1] = conv2vec(sg.semantic_branch.children())

	if cfg.debug:
		print(record.keys())
		l = []
		for k in record.keys():
			l.append(record[k][-1])
		print(l)

	if ind % 100 == 0 or ind == cfg.n_iter or cfg.debug:
		evaluator.aggregate()
		evaluator.summarize()
		num = min(4, label.shape[0])
		res = []
		for i in range(num):
			gen_seg = segs[-1][i].argmax(0)
			tarlabel = utils.tensor2label(label[i], cfg.n_class)
			genlabel = utils.tensor2label(gen_seg, cfg.n_class)
			image = (gen[i].clamp(-1, 1) + 1) / 2
			res.extend([image, genlabel, tarlabel])
		res = torch.cat([F.interpolate(m.float().unsqueeze(0), 256).cpu() for m in res])
		vutils.save_image(res, cfg.expr_dir + "/seg_%05d.png" % ind, nrow=3)

		res = [(gen[0].clamp(-1, 1) + 1) / 2]
		for i in range(len(segs)):
			gen_label = segs[i][0].argmax(0)
			gen_label_viz = utils.tensor2label(gen_label, cfg.n_class)
			res.append(gen_label_viz)
		res.append(utils.tensor2label(label[0], cfg.n_class))
		res = torch.cat([F.interpolate(m.float().unsqueeze(0), 256).cpu() for m in res])
		vutils.save_image(res, cfg.expr_dir + "/layer_%05d.png" % ind, nrow=3)
		
		utils.write_log(cfg.expr_dir, record)
		utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")

	if (ind % 1000 == 0 and ind > 0) or ind == cfg.n_iter:
		print("=> Snapshot model %d" % ind)
		evaluator.save(cfg.expr_dir + "/training_evaluation.npy")
		torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % ind)
		if cfg.trace_weight:
			np.save(cfg.expr_dir + "/trace_weight.npy", trace_weight[:ind])

os.system(f"python script/monitor.py --task log,seg,celeba-evaluator,agreement --gpu {cfg.gpu[0]} --model {cfg.expr_dir}")