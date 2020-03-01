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
data_dir = "record/celebahq1"
device = "cpu"
external_model = segmenter.get_segmenter(
    "celebahq", "checkpoint/faceparse_unet_512.pth", device=device)
label_list, cats = external_model.get_label_and_category_names()
label_list = [l[0] for l in label_list]
n_class = 16
metric = evaluate.DetectionMetric(n_class)
colorizer = utils.Colorize(n_class)
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if "." not in f]
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files = [f for f in model_files
    if "_linear" in f or "_generative" in f]
model_files.sort()
torch.manual_seed(20200301)
latents = torch.randn(len(model_files) * 4, 512).to(device)

model_path = f"checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_model_from_pth_file("stylegan", model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latents[0:1])
dims = [s.shape[1] for s in stage]


def get_output(generator, model_file, external_model, latent):
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image)
    dims = [s.shape[1] for s in stage]
    model_name = utils.listkey_convert(model_file,
        ["nonlinear", "linear", "generative", "spherical"])
    print(model_file)
    print(model_name)
    sep_model = get_semantic_extractor(model_name)(
        n_class=n_class,
        dims=dims).to(device)
    sep_model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        seg = sep_model(stage, True)[0]

    size = label.shape[2]
    image = F.interpolate(image, size=size, mode="bilinear")
    image = (image + 1) / 2
    res = [image]

    label_viz = colorizer(label[0]).float().unsqueeze(0) / 255.
    pred = seg.argmax(1)
    pred_viz = colorizer(pred).float().unsqueeze(0) / 255.
    res.extend([pred_viz, label_viz])
    return res


def process(res):
    res = torch.cat(res)
    res = F.interpolate(res, 256, mode="bilinear")
    return res


# get result from all models
paper_res = []
appendix_res = []
count = 0
for ind, model_file in enumerate(model_files):
    for i in range(2):
        latent = latents[i:i+1]
        paper_res.extend(get_output(
            generator, model_file, external_model, latent))
        count += 1
        #latent = latents[count:count+1]
        #appendix_res.extend(get_output(
        #    generator, model_file, external_model, latent))
        #count += 1

vutils.save_image(
    process(paper_res),
    f"qualitative_celeba_paper.png", nrow=6)
#vutils.save_image(
#    process(appendix_res),
#    f"qualitative_celeba_appendix.png", nrow=7)
