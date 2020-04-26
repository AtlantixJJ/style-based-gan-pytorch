import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
sys.path.insert(0, ".")

import model, utils
from model.semantic_extractor import get_semantic_extractor

WINDOW_SIZE = 100
n_class = 15
device = "cuda"
latent = torch.randn(1, 512, device=device)
colorizer = utils.Colorize(15)

# data
trace_path1 = "record/useless/l1_bce/celebahq_stylegan_linear_layer0,1,2,3,4,5,6,7,8_vbs16_l10.001_l1pos_-1/trace.npy"
trace_path2 = "record/useless/l1_bce/celebahq_stylegan_linear_layer0,1,2,3,4,5,6,7,8_vbs16_l10.0001_l1pos_-1/trace.npy"
trace1 = np.load(trace_path1) # (N1, 15, D)
trace2 = np.load(trace_path2) # (N2, 15, D)

# generator
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in trace_path1 else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latent)
image = (1 + image) / 2
dims = [s.shape[1] for s in stage]

sep_model1 = get_semantic_extractor("unit")(
    n_class=n_class,
    dims=dims).to(device)
sep_model1.weight.requires_grad = False
sep_model2 = get_semantic_extractor("unit")(
    n_class=n_class,
    dims=dims).to(device)
sep_model2.weight.requires_grad = False


# segmentation movie
os.system("rm video/*.png")
for ind in tqdm(range(trace1.shape[0])):
    sep_model1.weight.copy_(torch.from_numpy(trace1[ind]).unsqueeze(2).unsqueeze(2))
    sep_model2.weight.copy_(torch.from_numpy(trace2[ind]).unsqueeze(2).unsqueeze(2))
    label1 = sep_model1(stage)[0].argmax(1)
    label2 = sep_model2(stage)[0].argmax(1)
    label1_viz = colorizer(label1).unsqueeze(0) / 255.
    label2_viz = colorizer(label2).unsqueeze(0) / 255.
    diff_label = label2_viz.clone()
    for i in range(3):
        diff_label[:, i, :, :][label1 == label2] = 1
    imgs = [image, diff_label, label1_viz, label2_viz]
    imgs = [F.interpolate(x,
        size=256, mode="bilinear", align_corners=True) for x in imgs]
    vutils.save_image(torch.cat(imgs), "video/%04d.png" % ind, nrow=2)
os.system("ffmpeg -y -f image2 -r 12 -i video/%04d.png -pix_fmt yuv420p -b:v 16000k comparative_trace.mp4")