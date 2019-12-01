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
    latent_dir=rootdir+"dlatent_test",
    noise_dir=rootdir+"noise_test",
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
    # for all gpu
    gpu = args.gpu.split(",")
    for i, f in enumerate(files):
        basecmd = "python test.py --model %s --zero %d --gpu %s &"
        basecmd = basecmd % (osj(args.model, f), args.zero, gpu[i % len(gpu)])
        if i == len(gpu) - 1:
            basecmd = basecmd[:-2]
        os.system(basecmd)
    exit(0)

out_prefix = args.model.replace("expr/", "results/")

# Init model
cfg = config.config_from_name(args.model)
print(cfg)
generator = StyledGenerator(**cfg)
generator = generator.cuda()
generator.eval()

# Load model
model_files = glob.glob(args.model + "/*.model")
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

if args.zero:
    print("=> Use zero as noise")
    noise = [0] * 18
    for k in range(18):
        size = 4 * 2 ** (k // 2)
        noise[k] = torch.zeros(1, 1, size, size).cuda()
    generator.set_noise(noise)
else:
    generator.set_noise(None)

def func(latent, noise):
    with torch.no_grad():
        generator.set_noise(noise)
        gen, seg = generator.predict(latent)
        gen = (gen.clamp(-1, 1) + 1) / 2
        gen = gen.detach().cpu()
    return gen, seg

def evaluate_on_dataset(predict_func, ds, save_path="record.npy"):
    evaluator = utils.MaskCelebAEval(map_id=True)

    noise = None
    for ind, (latent_np, noise_np, image_np, label_np) in enumerate(tqdm(ds)):
        latent = torch.from_numpy(latent_np).float().cuda()
        if noise is None:
            noise = [torch.from_numpy(noise).float().cuda() for noise in noise_np]
        else:
            for i in range(len(noise)):
                noise[i] = torch.from_numpy(noise_np[i]).float().cuda()
        gen, seg = predict_func(latent, noise)
        if evaluator.map_id:
            label = evaluator.idmap(label_np)
        gen = gen[0]
        seg_np = seg[0].detach().cpu().numpy()
        score = evaluator.compute_score(seg_np, label)
        evaluator.accumulate(score)
        
        if ind < 4:
            image = torch.from_numpy(image_np).float()
            image = image.permute(2, 0, 1).unsqueeze(0) / 255.
            genlabel = utils.tensor2label(seg, ds.n_class).unsqueeze(0)
            tarlabel = utils.tensor2label(
                torch.from_numpy(label).unsqueeze(0),
                ds.n_class).unsqueeze(0)
            gen = gen.unsqueeze(0)
            res = [image, tarlabel, gen, genlabel]
            res = torch.cat([F.interpolate(item, (256, 256), mode="bilinear") for item in res])
            vutils.save_image(res, save_path.replace("record.npy", f"{ind}.png"),
                nrow=2, normalize=True, range=(0, 1), scale_each=True)

    evaluator.aggregate()
    evaluator.summarize()
    evaluator.save(save_path)

for model in model_files:
    print("=> Load from %s" % model)
    state_dict = torch.load(model, map_location='cuda:0')
    missed = generator.load_state_dict(state_dict, strict=False)
    print(missed)
    ind = model.rfind("/")
    iteration = model[ind+1:].replace("iter_", "").replace(".model", "")
    evaluate_on_dataset(func, ds, f"{out_prefix}_{args.zero}_{iteration}_record.npy")