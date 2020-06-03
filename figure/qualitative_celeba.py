import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
import torch.nn.functional as F
import model, utils, evaluate, segmenter
from model.semantic_extractor import get_semantic_extractor
from lib.netdissect.segviz import segment_visualization_single
from torchvision import utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="record/celebahq1", help="")
args = parser.parse_args()

# setup and constants
high_contrast = utils.CELEBA_COLORS
data_dir = args.dir
device = "cpu"
external_model = segmenter.get_segmenter(
    "celebahq", "checkpoint/faceparse_unet_512.pth", device=device)
label_list, cats = external_model.get_label_and_category_names()
label_list = np.array([l[0] for l in label_list])
n_class = 15
N_repeat = 2
metric = evaluate.DetectionMetric(ignore_classes=[0, 13], n_class=n_class)
colorizer = utils.Colorize(n_class)
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if "." not in f]
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files = [f for f in model_files
    if "_linear" in f or "_nonlinear" in f]
model_files.sort()
torch.manual_seed(1701)
latents = torch.randn(N_repeat, 512).to(device)

model_path = f"checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_model(model_path)
generator.to(device).eval()
with torch.no_grad():
    image, stage = generator.get_stage(latents[0:1])
dims = [s.shape[1] for s in stage]
noises = [generator.generate_noise(device=device) for _ in range(N_repeat)]
get_name = lambda x : utils.listkey_convert(x,
        ["nonlinear", "linear", "generative", "spherical", "unitnorm", "unit"],
        ["NSE-1", "LSE", "NSE-2", "LSE-F", "LSE-WF", "LSE-W"])

def get_classes(l, start=0):
    x = np.array(l)
    y = x.argsort()
    k = 0
    while x[y[k]] < 1e-3:
        k += 1
    y = y[k:][::-1]
    # all classes are the same
    names = label_list[y - 1 + start] 
    return x[y], names.tolist(), y + start


def get_output(generator, model_file, external_model, latent, noise):
    with torch.no_grad():
        generator.set_noise(noise)
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image)
    dims = [s.shape[1] for s in stage]
    model_name = utils.listkey_convert(model_file, ["nonlinear", "linear", "generative", "spherical", "unitnorm", "unit"])
    print(model_file)
    print(model_name)
    sep_model = get_semantic_extractor(model_name)(
        n_class=n_class,
        dims=dims).to(device)
    sep_model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        seg = sep_model(stage, True)[0]

    size = label.shape[2]
    image = F.interpolate(image,
        size=size, mode="bilinear", align_corners=True)
    image = (image + 1) / 2
    res = [image]

    label_viz = colorizer(label).float() / 255.
    pred = seg.argmax(1)
    pred_viz = colorizer(pred).float() / 255.
    res.extend([pred_viz, label_viz])

    for j in range(pred.shape[0]):
        metric(
            utils.torch2numpy(pred[j]),
            utils.torch2numpy(label[j]))
        gt, ct = metric.aggregate()
        metric.reset()

    return res, copy.deepcopy(gt), copy.deepcopy(ct)

def get_text(gt, ct):
    s = []
    vals, names, cats = get_classes(ct["IoU"])
    iou = gt["mIoU"]
    s.append([f"mIoU {iou:.3f}", high_contrast[0]])
    s.extend([[f"{name} {val:.3f}", high_contrast[cat]]
        for cat, name, val in zip(cats, names, vals)])
    return s[:7]

def process(res):
    res = torch.cat(res)
    res = F.interpolate(res, 256,
        mode="bilinear", align_corners=True)
    return utils.torch2numpy(res * 255).transpose(0, 2, 3, 1)


# get result from all models
paper_res = []
paper_text = []
count = 0
for ind, model_file in enumerate(model_files):
    for i in range(N_repeat):
        latent = latents[i:i+1]
        noise = noises[i]
        res, gt, ct = get_output(
            generator, model_file, external_model, latent, noise)
        paper_res.extend(res)
        paper_text.append(get_text(gt, ct))
        count += 1
paper_res = process(paper_res)

N_imgs = len(paper_res)
N_col = 6
N_row = N_imgs // N_col
imsize = 256
text_height = 33
pad = 5
CW = imsize + pad
CH = imsize + text_height
canvas_width = imsize * (N_col + 2) + pad * (N_col + 1)
canvas_height = (text_height + imsize) * N_row
canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
canvas.fill(255)

for idx, img in enumerate(paper_res):
    row, col = idx // N_col, idx % N_col
    col += 1
    model_name = get_name(model_files[row])
    # labeling
    sty = CH * row + int(text_height * 0.8)
    if col == 1:
        cv2.putText(canvas, model_name,
                (CW * col, sty),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                1, cv2.LINE_AA)
    elif row == 0 and col % 3 == 2:
        cv2.putText(canvas, f"Extracted",
                (CW * col, sty),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                1, cv2.LINE_AA)
    elif row == 0 and col % 3 == 0:
        cv2.putText(canvas, f"UNet-512",
                (CW * col, sty),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                1, cv2.LINE_AA)

    stx = text_height + CH * row
    edx = stx + imsize
    sty = CW * col - text_height
    edy = sty + imsize
    canvas[stx:edx, sty:edy] = img
    if idx % 3 == 0:
        idx = idx // 3
        delta = CW * (N_col + 1) if idx % 2 == 1 else 0
        for i, (text, rgb) in enumerate(paper_text[idx]):
            i += 1
            cv2.putText(canvas, text,
                (5 + delta, text_height + (idx // 2) * CH + 33 * i),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                2 if "mIoU" in text else 1, cv2.LINE_AA)


fig = plt.figure(figsize=(24, 8))
plt.imshow(canvas)
plt.axis("off")
plt.tight_layout()
plt.savefig(f"qualitative_celeba_paper.pdf", box_inches="tight")
plt.close()
