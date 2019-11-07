import os
from os.path import join as osj
import torch
import torch.nn.functional as F
import argparse
import glob
import numpy as np
import utils
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir, image_dir, seg_dir, map_class=[(3, 4), (5, 6), (7, 8)]):
        super(SegmentationDataset, self).__init__()
        self.map_class = map_class
        self.latent_dir = latent_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.latent_files = os.listdir(self.latent_dir)
        self.latent_files.sort()

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, index):
        name = self.latent_files[index]
        latent_path = osj(self.latent_dir, name)
        image_path = osj(self.image_dir, name.replace(".npy", ".jpg"))
        seg_path = osj(self.seg_dir, name.replace(".npy", ".png"))
        latent = np.load(latent_path)
        image = utils.imread(image_path).copy()
        label = utils.imread(seg_path).copy()
        if self.map_class is not None:
            for ct,cf in self.map_class:
                label[label == cf] = ct
        return latent, image, label

rootdir = "/home/xujianjin/data/datasets/CelebAMask-HQ/"
ds = SegmentationDataset(
    latent_dir=rootdir+"dlatent",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--step", type=int, default=8)
args = parser.parse_args()

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python test.py --model %s"
        basecmd = basecmd % osj(args.model, f)
        os.system(basecmd)
    exit(0)

out_prefix = args.model.replace("expr/", "results/")

cfg = config.config_from_name(args.model)
print(cfg)
generator = StyledGenerator(**cfg)
generator = generator.cuda()

model_files = glob.glob(args.model + "/*.model")
model_files.sort()
print("=> Load from %s" % model_files[-1])
missed = generator.load_state_dict(torch.load(
    model_files[-1], map_location='cuda:0'), strict=True)
print(missed)
generator.eval()

state_dict = torch.load("checkpoint/faceparse_unet.pth", map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.cuda()
faceparser.eval()
del state_dict

def compute_iou(a, b):
    if b.any() == False:
        return -1
    return (a & b).astype("float32").sum() / (a | b).astype("float32").sum()

def compute_score(seg, label, n=19):
    res = []
    for i in range(1, n):
        mask_dt = (seg == i)
        mask_gt = (label == i)
        score = compute_iou(mask_dt, mask_gt)
        res.append(score)
    return res

def aggregate(record):
    record["class_acc"] = [-1] * 19
    total = 0
    cnt = 0
    for i in range(1, 19):
        arr = np.array(record[i])
        arr = arr[arr > -1]
        cnt += arr.shape[0]
        total += arr.sum()
        record["class_acc"][i] = arr.mean()
    record["acc"] = total / cnt
    if "sigma" in record.keys():
        record["esd"] = np.array(record["sigma"]).mean()
    return record

def summarize(record):
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    print("=> Total accuracy: %.3f" % record["acc"])
    print("=> Class wise accuracy:")
    for i in range(1, 19):
        print("=> %s:\t%.3f" % (label_list[i - 1], record["class_acc"][i]))
    print("=> Image expected standard deviation: %.3f" % record["esd"])

tar_record = {i:[] for i in range(1,19)}
for latent, image, label in ds:
    image = torch.from_numpy(image).float().cuda()
    image = (image.permute(2, 0, 1) - 127.5) / 127.5
    with torch.no_grad():
        image_ = F.interpolate(image.unsqueeze(0), (512, 512))
        tar_seg = faceparser(image_)[0]
        tar_seg = tar_seg.argmax(0).detach().cpu().numpy()
    tar_score = compute_score(tar_seg, label)
    for i in range(1, 19):
        tar_record[i].append(tar_score[i - 1])
tar_record = aggregate(tar_record)
summarize(tar_record)
np.save("tar_record.npy", tar_record)