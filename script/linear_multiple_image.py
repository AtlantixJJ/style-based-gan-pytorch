"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset, loss
from model.linear import LinearSemanticExtractor, OVOLinearSemanticExtractor
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--train-size", default=4, type=int)
parser.add_argument(
    "--train-iter", default=200, type=int)
parser.add_argument(
    "--save-iter", default=50, type=int)
parser.add_argument(
    "--total-repeat", default=10, type=int)
parser.add_argument(
    "--test-dir", default="datasets/Synthesized_test")
parser.add_argument(
    "--total-class", default=16, type=int)
parser.add_argument(
    "--debug", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def fg_bg_idmap(x):
    return utils.idmap(x,
        n=args.total_class, # original label number
        map_from=list(range(2, args.total_class)),
        map_to=[1] * 14)

def hair_face_bg_idmap(x):
    return utils.idmap(x,
        n=args.total_class,
        map_from=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        map_to=  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1  ,0,  0,  0,  0,  0])

def full_idmap(x):
    return x

def bedroom_bed_idmap(x):
    return utils.idmap(x,
        n=args.total_class,
        map_from=[2, 3, 4],
        map_to=[0, 0, 0])

idmap = full_idmap
name = "full"


# constants setup
torch.manual_seed(65537)
device = "cuda" if int(args.gpu) > -1 else "cpu"
ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=None,
    seg_dir=args.data_dir + "/label")
dl = torch.utils.data.DataLoader(ds, batch_size=args.train_size, shuffle=False)

test_ds = dataset.LatentSegmentationDataset(
    latent_dir=args.test_dir + "/latent",
    noise_dir=args.test_dir + "/noise",
    image_dir=None,
    seg_dir=args.test_dir + "/label")
test_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# build model
print("=> Setup generator")
resolution = utils.resolution_from_name(args.model)
generator = StyledGenerator(resolution=resolution, semantic=f"conv-{args.total_class}-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
print(missing_dict)
generator.eval()

# set up input
latent = torch.randn(1, 512).to(device)
colorizer = utils.Colorize(args.total_class)
stylegan_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]


def test(generator, linear_model, test_dl):
    evaluator = evaluate.MaskCelebAEval()
    for i, sample in enumerate(tqdm(test_dl)):
        latent, noise, image, label = sample
        generator.set_noise(generator.parse_noise(noise[0].to(device)))
        image, seg = generator(latent.to(device))
        est_label = linear_model.predict(generator.stage)
        size = est_label.shape[2]
        
        seg = F.interpolate(seg, size=size, mode="bilinear")
        label = seg.argmax(1).detach().cpu()
        label = idmap(label)
        evaluator.calc_single(est_label, utils.torch2numpy(label))

        if args.debug == 1 and i > 4:
            break

    global_dic, class_dic = evaluator.aggregate()
    evaluator.summarize()
    return global_dic, class_dic


for ind, sample in enumerate(tqdm(dl)):
    if ind > args.total_repeat:
        break
    latents, noises, images, labels = sample
    labels = labels[:, :, :, 0]
    labels = idmap(labels)

    # Train a linear model based on train_size samples
    linear_model = LinearSemanticExtractor(args.total_class, stylegan_dims).to(device)
    for i in tqdm(range(args.train_iter)):
        # ensure we initialize different noise
        generator.set_noise(None)
        stages = []
        stage = []

        prev = cur = 0
        # equivalent to 1 iteration, in case memory is not sufficient
        for j in range(latents.shape[0]):
            generator(latents[j].to(device), seg=False)
            stages.append([s.detach() for s in generator.stage])
            cur += 1
            if (j + 1) % 4 != 0 and j + 1 != latents.shape[0]: # form a batch of 4
                continue
            for k in range(len(stages[0])):
                stage.append(torch.cat([s[k] for s in stages]))
            # optimization
            segs = linear_model(stage) # (N, C, H, W)
            segloss = loss.segloss(segs, labels[prev:cur].to(device))
            segloss.backward()
            prev = cur
            stages = []
            stage = []

        linear_model.optim.step()
        linear_model.optim.zero_grad()

        if (i + 1) % args.save_iter == 0 or (i + 1) == args.train_iter or args.debug == 1:
            fpath = f"results/linear_{ind}_i{i+1}_b{args.train_size}_idmap-{name}.model"
            torch.save(linear_model.state_dict(), fpath)
            global_dic, class_dic = test(generator, linear_model, test_dl)
            np.save(fpath.replace(".model", "_global.npy"), global_dic)
            np.save(fpath.replace(".model", "_class.npy"), class_dic)

        if i + 1 == args.train_iter or args.debug == 1:
            est_labels = segs[-1].argmax(1)
        
        if args.debug == 1 and i > 10:
            break

    image = generator(latents[:4, 0, :].to(device), seg=False)
    image = (1 + image.clamp(-1, 1).detach().cpu()) / 2
    est_labels = torch.from_numpy(linear_model.predict(generator.stage))
    est_labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in est_labels]
    labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels[:4]]
    res = []
    for img, lbl, pred in zip(image, labels_viz, est_labels_viz):
        res.extend([img.unsqueeze(0), lbl, pred])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest") for r in res]
    fpath = f"results/linear_train_{ind}_b{args.train_size}_idmap-{name}.png"
    vutils.save_image(torch.cat(res), fpath, nrow=3)

    labels = []
    latents = []

# model converges after 50 iterations, but train 100 iterations in default