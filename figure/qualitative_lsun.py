import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model, utils, evaluate, segmenter
from model.semantic_extractor import get_semantic_extractor
from lib.netdissect.segviz import segment_visualization_single
from torchvision import utils as vutils

# setup and constants
data_dir = "record/lsun"
device = "cuda"
external_model = segmenter.get_segmenter(
    "bedroom", device=device)
label_list, cats = external_model.get_label_and_category_names()
cg = utils.get_group(label_list)
cg_label = utils.get_group(label_list, False)
label_list = [l[0] for l in label_list]
object_metric = evaluate.DetectionMetric(
    n_class=cg[0][1] - cg[0][0])
material_metric = evaluate.DetectionMetric(
    n_class=cg[1][1] - cg[1][0])
n_class = 392
colorizer = lambda x: segment_visualization_single(x, 256)
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if ".npy" not in f]
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files.sort()
func = get_semantic_extractor("linear")
torch.manual_seed(65537)
latents = torch.randn(len(model_files) * 4, 512).to(device)

def get_output(generator, model_file, external_model, latent,
    flag=2):
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image)
        label = utils.torch2numpy(label)
    dims = [s.shape[1] for s in stage]
    sep_model = func(
        n_class=n_class,
        category_groups=cg,
        dims=dims).to(device)
    sep_model.load_state_dict(torch.load(model_file))

    with torch.no_grad():
        multi_segs = sep_model(stage)

    size = label.shape[2:]
    image = F.interpolate(image, size=size, mode="bilinear")
    image = (utils.torch2numpy(image[0]) + 1) * 127.5
    res = [image.transpose(1, 2, 0)]
    for i, seg in enumerate(multi_segs[:flag]):
        label_viz = colorizer(label[0, i])
        pred_group = segmenter.convert_multi_label(
            seg, cg_label, i)
        pred_group_viz = colorizer(pred_group)
        res.extend([pred_group_viz, label_viz])
    return res


def process(res):
    res = torch.from_numpy(np.stack(res))
    return res.permute(0, 3, 1, 2).float() / 255.


# get result from all models
paper_res = []
appendix_res = []
count = 0
for ind, model_file in enumerate(model_files):
    task = utils.listkey_convert(model_file, ["bedroom", "church"])
    model_name = utils.listkey_convert(
        model_file, ["stylegan2", "stylegan", "proggan"])
    model_path = f"checkpoint/{task}_lsun_256x256_{model_name}.pth"
    print(f"=> load {model_name} from {model_path}")
    generator = model.load_model_from_pth_file(
        model_name,
        model_path)
    generator.to(device).eval()

    for _ in range(2):
        latent = latents[count:count+1]
        paper_res.extend(get_output(
            generator, model_file, external_model, latent,
            flag=1))
        count += 1
        latent = latents[count:count+1]
        appendix_res.extend(get_output(
            generator, model_file, external_model, latent,
            flag=2))
        count += 1

vutils.save_image(
    process(paper_res),
    f"qualitative_lsun_paper.pdf", nrow=6)
vutils.save_image(
    process(appendix_res),
    f"qualitative_lsun_appendix.pdf", nrow=5)
