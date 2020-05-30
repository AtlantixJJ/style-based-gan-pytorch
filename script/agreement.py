"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj
from sklearn.metrics.pairwise import cosine_similarity
import evaluate, utils, config, dataset
from lib.face_parsing import unet
import model, segmenter
from lib.netdissect.segviz import segment_visualization_single
from model.semantic_extractor import get_semantic_extractor

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from weight_visualization import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--G", default="checkpoint/face_celebahq_1024x1024_stylegan.pth")
parser.add_argument("--gpu", default="0")
parser.add_argument("--norm", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


device = 'cuda' if int(args.gpu) > -1 else 'cpu'
cfg = 0
batch_size = 1
latent_size = 512
task = "celebahq"
colorizer = utils.Colorize(15) #label to rgb
model_path = args.G
generator = model.load_model(model_path).to(device)
model_path = "checkpoint/faceparse_unet_512.pth"
external_model = segmenter.get_segmenter(task, model_path, device)
labels, cats = external_model.get_label_and_category_names()
category_groups = utils.get_group(labels)
category_groups_label = utils.get_group(labels, False)
n_class = category_groups[-1][1]
utils.set_seed(65537)
latent = torch.randn(1, latent_size).to(device)
noise = False
op = getattr(generator, "generate_noise", None)
if callable(op):
    noise = op(device)

with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]

model_files = glob.glob(args.model + "/*extractor*.model")
model_files.sort()
if ".model" in args.model:
    model_files = [args.model]
latent = torch.randn(batch_size, latent_size, device=device)

def get_extractor_name(model_path):
    keywords = ["nonlinear", "linear", "spherical", "generative", "projective", "unitnorm", "unit"]
    for k in keywords:
        if k in model_path:
            return k


for model_file in model_files:
    res = []
    savepath = model_file.replace(".model", "")

    if noise:
        generator.set_noise(noise)

    layers = list(range(9))
    sep_model = 0
    if "layer" in model_file:
        ind = model_file.rfind("layer") + len("layer")
        s = model_file[ind:].split("_")[0]
        if ".model" in s:
            s = s.split(".")[0]
        layers = [int(i) for i in s.split(",")]
        cdim = np.array(dims)[layers].tolist()
        sep_model = get_semantic_extractor(get_extractor_name(model_file))(
            n_class=n_class,
            dims=cdim,
            use_bias="bias1" in model_file)
    sep_model.to(device).eval()
    print("=> Load from %s" % model_file)
    is_resize = "spherical" not in model_file
    state_dict = torch.load(model_file, map_location='cpu')
    missed = sep_model.load_state_dict(state_dict)
    evaluator = evaluate.MaskCelebAEval()
    for ind in tqdm.tqdm(range(3000)):
        with torch.no_grad():
            gen, stage = generator.get_stage(latent)
            gen = gen.clamp(-1, 1)

            stage = [s for i, s in enumerate(stage) if i in layers]
            if args.norm == "1":
                maxsize = max([s.shape[3] for s in stage])
                feat = torch.cat([utils.bu(s, maxsize) for s in stage], 1)
                feat = F.normalize(feat, 2, 1)
                stage = torch.split(feat, cdim, 1)
            est_label = sep_model(stage)[0].argmax(1)
            label = external_model.segment_batch(gen,
                resize=is_resize)
        label = F.interpolate(label.unsqueeze(1).float(),
            size=est_label.shape[2],
            mode="nearest").long()[:, 0, :, :]

        if ind == 4:
            res = torch.cat([utils.bu(r, 256).cpu() for r in res])
            vutils.save_image(res, savepath + "_seg.png", nrow=6)
        elif ind < 4:
            label_viz = colorizer(label).float() / 255.
            est_label_viz = colorizer(est_label).float() / 255.
            res.extend([(1 + gen) / 2, label_viz, est_label_viz])

        label = utils.torch2numpy(label)

        est_label = utils.torch2numpy(est_label).astype("int32")
        for j in range(batch_size):
            score = evaluator.calc_single(est_label[j], label[j])
        latent.normal_()


    evaluator.aggregate()
    clean_dic = evaluator.summarize()
    np.save(savepath + "_agreement", clean_dic)