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
    def __init__(self, latent_dir, image_dir, seg_dir):
        super(SegmentationDataset, self).__init__()
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
        image = utils.imread(image_path)
        segmentation = utils.imread(seg_path)
        return latent, image, segmentation

rootdir = "/home/xujianjin/data/datasets/CelebAMask-HQ/"
ds = SegmentationDataset(
    latent_dir=rootdir+"dlatent5000",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

latent, image, label = ds[10]
latent = torch.from_numpy(latent)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--step", type=int, default=8)
args = parser.parse_args()

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

gen = generator.g_synthesis(latent.unsqueeze(0).cuda())
gen = (gen.clamp(-1, 1) + 1) * 127.5
gen = gen.detach().cpu().numpy()
gen = gen[0].transpose(1, 2, 0)
seg_logit = generator.extract_segmentation()[-1]
seg_logit = F.interpolate(seg_logit, (512, 512), mode="bilinear")
seg_logit = seg_logit[0]
seg = seg_logit.argmax(dim=0).detach().cpu()
seg = seg.numpy()
for i in range(1, seg_logit.shape[0]):
    mask_dt = (seg == i)
    mask_gt = (label == i)
    score = compute_iou(mask_dt, mask_gt)
    print(score)
utils.imwrite("gen.png", gen)
utils.imwrite("seg_dt.png", utils.numpy2label(seg, 19))
utils.imwrite("seg_gt.png", utils.numpy2label(label, 19))

