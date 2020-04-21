import sys
sys.path.insert(0, ".")
import os
from os.path import join as osj
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse
import glob
import numpy as np
import utils, evaluate, dataset, config
from model.tf import Discriminator
from model.semantic_extractor import LinearSemanticExtractor
from lib.face_parsing import unet


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--recursive", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.recursive == "1":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files = [f for f in files if os.path.isdir(f"{args.model}/{f}")]
    files.sort()
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, f in enumerate(files):
        basecmd = "python script/test_segmentation.py --model %s --gpu %s"
        basecmd = basecmd % (osj(args.model, f), gpus[i % len(gpus)])
        slots[i % len(gpus)].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)

model_path = args.model
model = imsize = model_name = layers = 0
if "faceparse" in model_path:
    model = unet.unet(n_classes=15)
    model.load_state_dict(torch.load("checkpoint/faceparse_unet_512.pth"))
    model.cuda().eval()
    imsize = 512
    model_name = "faceparse"
    predict = lambda x: model(x)
elif "stylegandisc" in model_path:
    disc = Discriminator(resolution=1024)
    disc.load_state_dict(torch.load("checkpoint/karras2019stylegan-celebahq-1024x1024.for_d_basic.pt"))
    disc.cuda().eval()
    with torch.no_grad():
        d, stage = disc.get_stage(torch.randn(1, 3, 1024, 1024, device="cuda"))
    dims = [s.shape[1] for s in stage]
    layers = list(range(9))
    if "layer" in model_path:
        ind = model_path.rfind("layer") + len("layer")
        layers = [int(i) for i in model_path[ind:].split(",")]
        dims = np.array(dims)[layers].tolist()
    print(dims)
    imsize = 1024
    model_name = "stylegandisc"
    model = LinearSemanticExtractor(
        n_class=15, dims=dims)
    model_file = glob.glob(f"{model_path}/*.model")
    model_file.sort()
    model.load_state_dict(torch.load(model_file[0]))
    model.cuda().eval()

    def predict(x):
        d, stage = disc.get_stage(x)
        stage = [s for i, s in enumerate(stage) if i in layers]
        seg = model(stage)[0]
        if seg.shape[2] != x.shape[2]:
            seg = F.interpolate(seg,
                size=x.shape[2], mode="bilinear", align_corners=True)
        return seg

print(model_path)

rootdir = "../datasets/CelebAMask-HQ/"
ds = dataset.ImageSegmentationDataset(
    root=rootdir,
    size=imsize,
    image_dir=f"CelebA-HQ-img",
    label_dir=f"CelebAMask-HQ-mask-15",
    file_list=f"{rootdir}/test.list")
dl = torch.utils.data.DataLoader(ds, batch_size=1)
result = []
colorizer = utils.Colorize(15)
evaluator = evaluate.MaskCelebAEval()
for ind, (x, y) in enumerate(tqdm(dl)):
    x = x.cuda()
    y = y[:, 0, :, :, 0]
    with torch.no_grad():
        tar_seg = predict(x)
    est_label = tar_seg.argmax(1)
    for i in range(tar_seg.shape[0]):
        evaluator.calc_single(
            utils.torch2numpy(est_label[i]),
            utils.torch2numpy(y[i]))

    if ind < 4:
        label_viz = colorizer(y).unsqueeze(0).float() / 255.
        est_label_viz = colorizer(est_label)
        est_label_viz = est_label_viz.unsqueeze(0).float() / 255.
        image = (x.clamp(-1, 1) + 1) / 2
        result.extend([image, label_viz, est_label_viz])

test_images = [F.interpolate(img.detach().cpu(), size=256, mode="nearest")
    for img in result]
vutils.save_image(torch.cat(test_images),
    f"{model_path}_test.png", nrow=3)

evaluator.aggregate()
np.save(f"{model_path}_agreement.npy", evaluator.summarize())
