import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model, utils, evaluate, segmenter
from model.semantic_extractor import get_semantic_extractor, get_extractor_name
from lib.netdissect.segviz import segment_visualization_single
from lib.netdissect.segviz import high_contrast
from torchvision import utils as vutils
import cv2

# setup and constants
model_type = sys.argv[1]
data_dir = "record/lsun"
device = "cuda"
external_model = segmenter.get_segmenter(
    "bedroom", device=device)
label_list, cats = external_model.get_label_and_category_names()
cg = utils.get_group(label_list)
cg_label = utils.get_group(label_list, False)
label_list = np.array([l[0] for l in label_list])
object_metric = evaluate.DetectionMetric(
    n_class=cg[0][1] - cg[0][0])
material_metric = evaluate.DetectionMetric(
    n_class=cg[1][1] - cg[1][0])
metrics = [object_metric, material_metric]
n_class = 392
colorizer = lambda x: segment_visualization_single(x, 256)
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if os.path.isdir(f) and model_type in f]
print(model_files)
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files.sort()
torch.manual_seed(1005)
latents = torch.randn(len(model_files) * 4, 512).to(device)

def get_classes(l, start=0):
    x = np.array(l)
    y = x.argsort()
    k = 0
    while x[y[k]] < 1e-3:
        k += 1
    y = y[k:][::-1]
    # all classes are the same
    names = label_list[y + start] 
    return x[y], names.tolist(), y + start

def get_output(generator, model_file, external_model, latent,
    flag=2):
    with torch.no_grad():
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image)
        label = utils.torch2numpy(label)
    dims = [s.shape[1] for s in stage]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        category_groups=cg,
        dims=dims).to(device)
    sep_model.load_state_dict(torch.load(model_file))

    with torch.no_grad():
        multi_segs = sep_model(stage)

    size = label.shape[2:]
    image = F.interpolate(image,
        size=size, mode="bilinear", align_corners=True)
    image = (utils.torch2numpy(image[0]) + 1) * 127.5
    res = [image.transpose(1, 2, 0)]
    for i, seg in enumerate(multi_segs[:flag]):
        label_viz = colorizer(label[0, i])
        pred_group = segmenter.convert_multi_label(
            seg, cg_label, i)
        pred_group_viz = colorizer(pred_group)
        res.extend([pred_group_viz, label_viz])
        #
        # evaluate
        l = label[:, i, :, :] - cg_label[i][0]
        l[l<0] = 0

        # collect training statistic
        est_label = seg[-1].argmax(1)
        gts = []
        cts = []
        for j in range(est_label.shape[0]):
            metrics[i](
                utils.torch2numpy(est_label[j]),
                utils.torch2numpy(l[j]))
            gt, ct = metrics[i].aggregate()
            gts.append(copy.deepcopy(gt))
            cts.append(copy.deepcopy(ct))
        metrics[i].reset()

    return res, gts, cts

def get_text(gts, cts):
    s = []
    for i in range(len(gts)):
        vals, names, cats = get_classes(cts[i]["IoU"])
        iou = gts[i]["mIoU"]
        s.append([f"mIoU {iou:.3f}", high_contrast[0]])
        s.extend([[f"{name} {val:.3f}", high_contrast[cat]]
            for cat, name, val in zip(cats, names, vals)])
    return s[:7]

# get result from all models
paper_res = []
appendix_res = []
paper_texts = []
appendix_texts = []
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
        res, gts, cts = get_output(
            generator, model_file, external_model, latent,
            flag=1)
        paper_res.extend(res)
        paper_texts.append(get_text(gts, cts))
        count += 1

        latent = latents[count:count+1]
        res, gts, cts = get_output(
            generator, model_file, external_model, latent,
            flag=2)
        appendix_res.extend(res)
        appendix_texts.append(get_text(gts, cts))
        count += 1

N_imgs = len(paper_res)
N_col = 6
N_row = N_imgs // N_col
imsize = 256
canvas_width = imsize * (N_col + 2)
canvas_height = imsize * N_row
canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
canvas.fill(255)

for idx, img in enumerate(paper_res):
    row, col = idx // N_col, idx % N_col
    col += 1
    canvas[imsize * row : imsize * (row + 1),
            imsize * col : imsize * (col + 1)] = img
    if idx % 3 == 0:
        idx = idx // 3
        delta = imsize * (N_col + 1) if idx % 2 == 1 else 0
        for i, (text, rgb) in enumerate(paper_texts[idx]):
            i += 1
            cv2.putText(canvas, text,
                (5 + delta, idx // 2 * imsize + 33 * i),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                2 if "mIoU" in text else 1, cv2.LINE_AA)

fig = plt.figure(figsize=(30, 15))
plt.imshow(canvas)
plt.axis("off")
plt.tight_layout()
plt.savefig(f"qualitative_lsun_paper.pdf", box_inches="tight")
plt.close()