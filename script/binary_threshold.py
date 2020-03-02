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
import model, segmenter, evaluate
from lib.netdissect.segviz import segment_visualization_single
from model.semantic_extractor import get_semantic_extractor


data_dir = "record/l1"
device = "cuda"
batch_size = 1
latent_size = 512
latent = torch.randn(batch_size, latent_size, device=device)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--test-dir", default="../datasets/Synthesized_test")
parser.add_argument(
    "--test-size", default=256, type=int)
parser.add_argument(
    "--T", default=0.01, type=float)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])


# test data
test_ds = dataset.LatentSegmentationDataset(
    latent_dir=args.test_dir + "/latent",
    noise_dir=args.test_dir + "/noise",
    image_dir=None,
    seg_dir=args.test_dir + "/label")
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)


# find all models
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if os.path.isdir(f)]
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files = [f for f in model_files if "_linear" in f]
model_files.sort()
with open("tmp", "w") as f:
    f.write("\n".join(model_files))

# init
external_model = segmenter.get_segmenter(
    "celebahq", "checkpoint/faceparse_unet_512.pth", device=device)
model_path = f"checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_model_from_pth_file("stylegan", model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]
label_list, cats = external_model.get_label_and_category_names()
label_list = [l[0] for l in label_list]
n_class = len(label_list) + 1
metric = evaluate.DetectionMetric(n_class)
sep_model = get_semantic_extractor("linear")(
    n_class=n_class,
    dims=dims)
sep_model.to(device).eval()


def evaluation(generator, sep_model, test_dl):
    evaluator = evaluate.MaskCelebAEval()
    for i, sample in enumerate(test_dl):
        latent, noise, image, label = sample
        label = label[:, :, :, 0]
        generator.set_noise(generator.parse_noise(noise[0].to(device)))
        with torch.no_grad():
            image, stage = generator.get_stage(latent[0].to(device))
            est_label = sep_model.predict(stage) 
        evaluator.calc_single(est_label, utils.torch2numpy(label))
            
        if i >= args.test_size:
            break

    global_dic, class_dic = evaluator.aggregate()
    return global_dic["mIoU"]


def small_absolute(x, margin=0.05):
    x[(x<margin)&(x>-margin)]=0
    return x


def surgery(state_dict, margin):
    for k in state_dict.keys():
        state_dict[k] = small_absolute(
            state_dict[k], margin)


def func(margin):
    state_dict = copy.deepcopy(origin_state_dict)
    surgery(state_dict, margin)
    sep_model.load_state_dict(state_dict)
    mIoU = evaluation(generator, sep_model, test_dl)
    val = (1 - args.T) * original_mIoU - mIoU

    print("=> margin=%.5f mIoU=%.5f original=%.5f val=%.5f" %
        (margin, mIoU, original_mIoU, val))
    return val

# func is increasing function
def bin_search(TOLERANCE=1e-4):
    left = 0.0
    right = 0.1
    while right - left > TOLERANCE:
        mid = (left + right) / 2
        res = func(mid)
        if res <= 0: #
            left = mid
        else:
            right = mid
    return left


for ind, model_file in enumerate(model_files):
    sep_model = get_semantic_extractor("linear")(
        n_class=n_class,
        dims=dims).to(device)
    origin_state_dict = torch.load(model_file)
    sep_model.load_state_dict(origin_state_dict)
    original_mIoU = evaluation(generator, sep_model, test_dl)
    threshold = bin_search()
    with open(model_file.replace("/stylegan_linear_extractor.model", "threshold.txt"), "w") as f:
        f.write(str(threshold))