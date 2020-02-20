import torch, numpy, os, argparse, sys, shutil
sys.path.insert(0, ".")
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing import unet
import config
import utils, evaluate, loss, model

from lib.netdissect.segviz import segment_visualization
from lib.netdissect.segmenter import UnifiedParsingSegmenter
from lib.netdissect import proggan
from lib.netdissect.zdataset import standard_z_sample, z_dataset_for_model

cfg = config.FixSegConfig()
cfg.parse()
s = str(cfg)
cfg.setup()
print(s)
        
latent = torch.randn(1, 512).to(cfg.device)
generator = proggan.from_pth_file(cfg.load_path).to(cfg.device)
generator.eval()
image, stage = generator.get_stage(latent, True)
dims = [s.shape[1] for s in stage]
#z_dataset = z_dataset_for_model(generator, size=cfg.n_iter)
#with torch.no_grad():
#    z_loader = torch.utils.data.DataLoader(z_dataset,
#                    batch_size=cfg.batch_size, num_workers=2,
#                    pin_memory=True)
segmenter = UnifiedParsingSegmenter()
labels, cats = segmenter.get_label_and_category_names()
with open(cfg.expr_dir + "/config.txt", "w") as f:
    f.write(s)
    for i, (label, cat) in enumerate(labels):
        f.write('%s %s\n' % (label, cat))


def get_group(labels):
    prev_cat = labels[0][1]
    prev_idx = 0
    cur_idx = 0
    groups = []

    for label, cat in labels:
        if cat != prev_cat:
            cur_idx += 1 # plus one for unlabeled class
            groups.append([prev_idx, cur_idx])
            prev_cat = cat
            prev_idx = cur_idx
        cur_idx += 1
    groups.append([prev_idx, cur_idx + 1])
    return groups


category_groups = get_group(labels)
n_class = category_groups[-1][1]

seg = segmenter.segment_batch(image)
linear_model = model.linear.LinearSemanticExtractor(
    n_class, dims, None, category_groups).to(cfg.device)
output = linear_model(stage)

for ind in tqdm(range(cfg.n_iter)):
	ind += 1
	latent.normal_()

	with torch.no_grad(): # fix main network
		gen, stage = generator.get_stage(latent, seg=False, detach=True)
		label = segmenter.segment_batch(gen).argmax(1)

	multi_segs = linear_model(stage)
    segloss = 0
    for i,segs in enumerate(multi_segs):
	    segloss = segloss + loss.segloss(segs, label[])

	regloss = 0
	if cfg.ortho_reg > 0:
		ortho_loss = ortho_convs(sg.semantic_branch.children())
		regloss = regloss + cfg.ortho_reg * ortho_loss
	if cfg.positive_reg > 0:
		positive_loss = positive_convs(sg.semantic_branch.children())
		regloss = regloss + cfg.positive_reg * positive_loss

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
		scores = evaluator.calc_single(gen_label_np[i], label_np[i])

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