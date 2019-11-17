import os
from os.path import join as osj
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import argparse
import glob
from tqdm import tqdm
import numpy as np
import utils
import dataset
from model.tfseg import StyledGenerator
import config
from lib.face_parsing import unet

rootdir = "datasets/CelebAMask-HQ/"
ds = dataset.LatentSegmentationDataset(
    latent_dir=rootdir+"dlatent",
    image_dir=rootdir+"CelebA-HQ-img",
    seg_dir=rootdir+"CelebAMask-HQ-mask")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--zero", type=int, default=0, help="Use zero as noise")
parser.add_argument("--gpu", default="0")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python test.py --model %s --zero %d"
        basecmd = basecmd % (osj(args.model, f), args.zero)
        os.system(basecmd)
    exit(0)

out_prefix = args.model.replace("expr/", "results/")

# Init model
cfg = config.config_from_name(args.model)
print(cfg)
generator = StyledGenerator(**cfg)
generator = generator.cuda()

# Load model
model_files = glob.glob(args.model + "/*.model")
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)
print("=> Load from %s" % model_files[-1])
missed = generator.load_state_dict(torch.load(
    model_files[-1], map_location='cuda:0'), strict=True)
print(missed)
generator.eval()

if args.zero:
    print("=> Use zero as noise")
    noise = [0] * 18
    for k in range(18):
        size = 4 * 2 ** (k // 2)
        noise[k] = torch.zeros(1, 1, size, size).cuda()
    generator.set_noise(noise)

evaluator = utils.MaskCelebAEval(map_id=True)

for i, (latent_np, image_np, label_np) in enumerate(tqdm(ds)):
    latent = torch.from_numpy(latent_np).unsqueeze(0).float().cuda()
    with torch.no_grad():
        gen, seg = generator.predict(latent)
        gen = (gen.clamp(-1, 1) + 1) / 2
        gen = gen.detach().cpu()
    if evaluator.map_id:
        label = evaluator.idmap(label_np)
    gen = gen[0]
    seg = seg[0].detach().cpu().numpy()
    score = evaluator.compute_score(seg, label)
    evaluator.accumulate(score)
    
    if i < 4:
        image = torch.from_numpy(image_np).float()
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.
        genlabel = utils.numpy2label(seg, ds.n_class).transpose(2, 0, 1)
        genlabel = torch.from_numpy(genlabel).float().unsqueeze(0)
        gen = gen.unsqueeze(0)
        res = [image, genlabel, gen]
        res = torch.cat([F.interpolate(item, (256, 256), mode="bilinear") for item in res])
        vutils.save_image(res, f"{out_prefix}_{i}.png",
            nrow=2, normalize=True, range=[0, 1], scale_each=True)

evaluator.aggregate()
evaluator.summarize()
evaluator.save(f"{out_prefix}_record.npy")