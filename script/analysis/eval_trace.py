import sys, os, argparse
sys.path.insert(0, ".")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--trace", default="record/")
parser.add_argument(
    "--segment", default=-1, type=int)
parser.add_argument(
    "--n-segment", default=-1, type=int)
parser.add_argument(
    "--gpu", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args = parser.parse_args()

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import model, evaluate, utils, dataset
from model.semantic_extractor import get_semantic_extractor


WINDOW_SIZE = 100
n_class = 15
device = "cuda"
trace_path = args.trace

latent = torch.randn(1, 512, device=device)
colorizer = utils.Colorize(15)

# generator
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in trace_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
image = (1 + image) / 2
dims = [s.shape[1] for s in stage]

sep_model = get_semantic_extractor("unit")(
    n_class=n_class,
    dims=dims).to(device)
sep_model.weight.requires_grad = False

# data
trace = np.load(trace_path) # (N, 15, D)
if args.n_segment > 0:
    step = int(float(trace.shape[0]) / args.n_segment)
    trace = trace[step * args.segment : step * (args.segment + 1)]

layers = list(range(9))
if "layer" in trace_path:
    ind = trace_path.rfind("layer") + len("layer")
    s = trace_path[ind:].split("_")[0]
    layers = [int(i) for i in s.split(",")]
    dims = np.array(dims)[layers].tolist()
evaluators = [evaluate.MaskCelebAEval() for _ in range(trace.shape[0])]
ds = dataset.LatentSegmentationDataset(
    latent_dir="datasets/Synthesized/latent",
    noise_dir="datasets/Synthesized/noise",
    image_dir="datasets/Synthesized/image",
    seg_dir="datasets/Synthesized/label")
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

for ind, sample in enumerate(dl):
    latents, noises, images, labels = sample
    latents = latents.squeeze(1).to(device)
    labels = labels[:, :, :, 0].long().unsqueeze(1)
    labels = utils.torch2numpy(labels)
    with torch.no_grad():
        gen, stage = generator.get_stage(latents)
        stage = [s for i, s in enumerate(stage) if i in layers]

    for i in tqdm(range(trace.shape[0])):
        sep_model.weight.copy_(torch.from_numpy(trace[i]).unsqueeze(2).unsqueeze(2))
        est_label = sep_model.predict(stage)
        #label = external_model.segment_batch(gen, resize=is_resize)
        for j in range(est_label.shape[0]):
            evaluators[i].calc_single(est_label[j], labels[j])

    dics = []
    for i in range(trace.shape[0]):
        evaluators[i].aggregate()
        dics.append(evaluators[i].summarize())
        name = f"eval_trace"
        if args.n_segment > 0:
            name = f"{name}_{args.segment}"
        np.save(name, dics)