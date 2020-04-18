import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
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
model_path = f"checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_model_from_pth_file("stylegan", model_path)
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
weight = trace[-1]

# segmentation movie
img = F.interpolate(image, size=256, mode="bilinear")
for i in range(trace.shape[0]):
    sep_model.weight.copy_(torch.from_numpy(trace[i]).unsqueeze(2).unsqueeze(2))
    seg = sep_model(stage)[0]
    label_viz = colorizer(seg.argmax(1)).unsqueeze(0) / 255.
    label_viz = F.interpolate(label_viz, size=256, mode="bilinear")
    vutils.save_image(torch.cat([img, label_viz]), "video/%04d.png" % i)
os.system("ffmpeg -y -f image2 -r 12 -i video/%04d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")


# initialization
moving_avg_trace = np.zeros_like(trace)
step_delta = np.zeros_like(trace)



# variance
weight_var = trace.std(0)
fig = plt.figure(figsize=(12, 12))
maximum, minimum = weight_var.max(), weight_var.min()
for j in range(15):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(weight_var[j])
    for x in segments:
        ax.axvline(x=x)
    ax.axes.get_xaxis().set_visible(False)
    #ax.set_ylim([weight[j].min(), weight[j].max()])
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_trace_var.png", bbox_inches='tight')
plt.close()


# weight vector
fig = plt.figure(figsize=(12, 12))
maximum, minimum = weight.max(), weight.min()
for j in range(15):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(weight[j])
    for x in segments:
        ax.axvline(x=x)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_trace_weight.png", bbox_inches='tight')
plt.close()

for i in range(trace.shape[0]):
    bg, ed = max(0, i - WINDOW_SIZE // 2), min(i + WINDOW_SIZE // 2, trace.shape[0])
    moving_avg_trace[i] = trace[bg:ed].mean(0)

step_delta[1:] = trace[1:] - trace[:-1]
step_delta[0] = 0
step_delta_norm = np.linalg.norm(step_delta, ord=2, axis=1)

fig = plt.figure(figsize=(10, 10))
maximum, minimum = step_delta_norm.max(), step_delta_norm.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(step_delta_norm[:, j])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_trace_stepdeltanorm.png", bbox_inches='tight')
plt.close()

moving_avg_delta = trace - moving_avg_trace
moving_avg_delta_norm = np.linalg.norm(moving_avg_delta, ord=2, axis=1)

fig = plt.figure(figsize=(10, 10))
maximum, minimum = moving_avg_delta_norm.max(), moving_avg_delta_norm.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(moving_avg_delta_norm[:, j])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_trace_movingavgdeltanorm.png", bbox_inches='tight')
plt.close()


