import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model, utils, evaluate, segmenter
from model.semantic_extractor import *
from lib.netdissect.segviz import segment_visualization_single
from lib.netdissect.segviz import high_contrast
from torchvision import utils as vutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="proggan", help="")
parser.add_argument("--dataset", default="bedroom", help="")
parser.add_argument("--N", default=2, type=int)
args = parser.parse_args()

# setup and constants
dataset = args.dataset
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
model_files = [f for f in model_files
    if os.path.isdir(f) and dataset in f and args.model in f]
print(model_files)
model_files = [glob.glob(f"{f}/*.model")[0] for f in model_files]
model_files.sort()
torch.manual_seed(1005)
latents = torch.randn(args.N, 512).to(device)
get_name = lambda x : utils.listkey_convert(x,
        ["nonlinear", "linear", "generative", "spherical", "unitnorm", "unit"],
        ["NSE-1", "LSE", "NSE-2", "LSE-F", "LSE-WF", "LSE-W"])

def get_classes(l, start=0):
    x = np.array(l)
    y = x.argsort()
    k = 0
    while x[y[k]] < 0.1:
        k += 1
        if k >= x.shape[0]:
            return [], [], []
    y = y[k:][::-1]
    # all classes are the same
    names = label_list[y + start] 
    return x[y], names.tolist(), y + start

def get_output(stage, sep_model, flag=2):
    with torch.no_grad():
        multi_segs = sep_model(stage)

    res = [[], []]
    gts = [[], []]
    cts = [[], []]
    for i, seg in enumerate(multi_segs[:flag]):
        pred_group = segmenter.convert_multi_label(seg, cg_label, i)
        pred_group_viz = colorizer(pred_group)

        res[i].append(pred_group_viz)

        # evaluate
        l = label[:, i, :, :] - cg_label[i][0]
        l[l<0] = 0

        # collect training statistic
        est_label = seg[-1].argmax(1)
        for j in range(est_label.shape[0]):
            metrics[i](
                utils.torch2numpy(est_label[j]),
                utils.torch2numpy(l[j]))
            gt, ct = metrics[i].aggregate()
            gts[i].append(copy.deepcopy(gt))
            cts[i].append(copy.deepcopy(ct))
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

model_path = f"checkpoint/{dataset}_lsun_256x256_{args.model}.pth"
print(f"=> load {args.model} from {model_path}")
generator = model.load_model(model_path)
generator.to(device).eval()

# get result from all models
paper_res = [[], []]
appendix_res = [[], []]
paper_texts = [[], []]
appendix_texts = [[], []]
count = 0
images = []
for i in range(args.N):
    with torch.no_grad():
        image, stage = generator.get_stage(latents[i:i+1])
        image = image.clamp(-1, 1)
        label = external_model.segment_batch(image)
        image = 127.5 * (image + 1)
        image = utils.torch2numpy(image[0].permute(1, 2, 0))
        label = utils.torch2numpy(label)
    dims = np.array([s.shape[1] for s in stage])
    images.append(image)
    for j in range(2):
        paper_res[j].append(colorizer(label[0, j]))
        
    for ind, model_file in enumerate(model_files):
        layers = get_layers(model_file)
        sep_model = get_semantic_extractor(get_extractor_name(model_file))(
            n_class=n_class,
            category_groups=cg,
            dims=list(dims[layers[:dims.shape[0]]])).to(device)
        sep_model.load_state_dict(torch.load(model_file))

        res, gts, cts = get_output(
            stage, sep_model, flag=2)
        for j in range(2):
            paper_res[j].extend(res[j])
            paper_texts[j].append(get_text(gts[j], cts[j]))
        count += 1

        """
        latent = latents[count:count+1]
        res, gts, cts = get_output(
            generator, model_file, external_model, latent,
            flag=2)
        appendix_res.extend(res)
        appendix_texts.append(get_text(gts, cts))
        count += 1
        """
    

paper_res = paper_res[0] + paper_res[1]
paper_texts = paper_texts[0] + paper_texts[1]
for r in paper_res: print(r.shape)

N_imgs = len(paper_res)
N_col = 1 + len(model_files)
N_row = args.N * 2
imsize = 256
pad = 5
text_height = 33
CH, CW = imsize * 2, imsize + pad
canvas_width = CW * (N_col + 2)
canvas_height = text_height + CH * N_row
canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
canvas.fill(255)
heads = ["UPerNet"] + [get_name(m) for m in model_files]
for idx, img in enumerate(paper_res):
    row, col = idx // N_col, idx % N_col
    if row == 0:
        sty = int(text_height * 0.8)
        cv2.putText(canvas, heads[col],
                (CW * col + imsize // 3, sty),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                1, cv2.LINE_AA)

    if col == 0: # put image in below
        stx = text_height + CH * row + imsize
        sty = 0
        edx = stx + imsize
        edy = sty + imsize
        canvas[stx:edx, sty:edy] = images[row % args.N]

    stx = text_height + CH * row
    sty = CW * col
    edx = stx + imsize
    edy = sty + imsize
    canvas[stx:edx, sty:edy] = img

    if col >= 1:
        didx = idx - (row + 1) * 2
        for i, (text, rgb) in enumerate(paper_texts[didx]):
            i += 1
            cv2.putText(canvas, text,
                (5 + CW * col, row * CH + imsize + text_height * (i+1)),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                2 if "mIoU" in text else 1, cv2.LINE_AA)

fig = plt.figure(figsize=(30, 15))
plt.imshow(canvas)
plt.axis("off")
plt.tight_layout()
plt.savefig(f"qualitative_lsun_{dataset}_{args.model}_paper.pdf", box_inches="tight")
plt.close()