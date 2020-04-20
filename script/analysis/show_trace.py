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
name = sys.argv[1] # fixseg_1.0_mul-16
name = name.replace("expr/", "")
trace_path = f"expr/{name}/trace.npy"
latent = torch.randn(1, 512, device=device)
colorizer = utils.Colorize(15)

# generator
#model_path = f"checkpoint/face_celebahq_1024x1024_stylegan.pth"
model_path = f"checkpoint/face_ffhq_1024x1024_stylegan2.pth"
generator = model.load_model_from_pth_file("stylegan2", model_path)
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
w = trace[100]


# segmentation movie
img = F.interpolate(image, size=256, mode="bilinear")
os.system("rm video/*.png")
for i in tqdm(range(trace.shape[0])):
    sep_model.weight.copy_(torch.from_numpy(trace[i]).unsqueeze(2).unsqueeze(2))
    seg = sep_model(stage)[0]
    label_viz = colorizer(seg.argmax(1)).unsqueeze(0) / 255.
    label_viz = F.interpolate(label_viz, size=256, mode="bilinear")
    vutils.save_image(torch.cat([img, label_viz]), "video/%04d.png" % i)
os.system("ffmpeg -y -f image2 -r 12 -i video/%04d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")