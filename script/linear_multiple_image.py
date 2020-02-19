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
    "--train-iter", default=100, type=int)
parser.add_argument(
    "--test-size", default=1000, type=int)
parser.add_argument(
    "--total-class", default=16, type=int)
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
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# build model
print("=> Setup generator")
resolution = utils.resolution_from_name(args.model)
generator = StyledGenerator(resolution=resolution, semantic=f"conv-{args.total_class}-1").to(device)
state_dict = torch.load(args.model, map_location='cpu')
missing_dict = generator.load_state_dict(state_dict, strict=False)
print(missing_dict)
generator.eval()

# setup test
print("=> Setup test data")
test_latents = torch.randn(args.test_size, 512)
test_noises = [generator.generate_noise() for _ in range(args.test_size)]

# set up input
latent = torch.randn(1, 512).to(device)
colorizer = utils.Colorize(args.total_class)
stylegan_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]


test_size = args.test_size
test_latents = torch.randn(test_size, 512)
test_noises = [generator.generate_noise() for _ in range(test_size)]


def test(generator, linear_model, test_latents, test_noises, N):
    result = []
    evaluator = evaluate.MaskCelebAEval()
    for i in tqdm(range(N)):
        latent = test_latents[i:i+1].to(device)
        noise = [n.to(latent.device) for n in test_noises[i]]
        generator.set_noise(noise)
        image, seg = generator(latent)
        est_label = linear_model.predict(generator.stage)
        size = est_label.shape[2]
        
        seg = F.interpolate(seg, size=size, mode="bilinear")
        label = seg.argmax(1).detach().cpu()
        label = idmap(label)
        evaluator.calc_single(est_label, utils.torch2numpy(label))

        if i < 8:
            label_viz = colorizer(label).unsqueeze(0).float() / 255.
            est_label_viz = torch.from_numpy(colorizer(est_label))
            est_label_viz = est_label_viz.permute(2, 0, 1).unsqueeze(0).float() / 255.
            image = (image.detach().cpu().clamp(-1, 1) + 1) / 2
            result.extend([image, label_viz, est_label_viz])

    global_dic, class_dic = evaluator.aggregate()
    evaluator.summarize()
    return global_dic, class_dic, result


images = []
latents = []
labels = []
for ind, sample in enumerate(tqdm(dl)):
    latent, noise, image, label = sample
    label = label[:, :, :, 0].unsqueeze(0)
    label = idmap(label)
    labels.append(label)
    latents.append(latent)

    if (ind + 1) % args.train_size != 0:
        continue

    labels = torch.cat(labels).to(device)
    latents = torch.cat(latents).to(device)
    labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels]
    est_labels = 0
    images = []

    # Train a linear model
    linear_model = LinearSemanticExtractor(args.total_class, stylegan_dims).to(device)
    for i in tqdm(range(args.train_iter)):
        # ensure we initialize different noise
        generator.set_noise(None)
        stages = []
        stage = []

        # equivalent to batch_size=1, in case memory is not sufficient
        for latent in latents:
            image = generator(latent, seg=False)
            if i + 1 == args.train_iter:
                img = image.clamp(-1, 1).detach().cpu()
                images.append((1 + img) / 2)
            stages.append([s.detach() for s in generator.stage])
        for i in range(len(stages)):
            stage.append(torch.cat([s[i] for s in stages]))
        
        # optimization
        segs = linear_model(stage) # (N, C, H, W)
        segloss = loss.segloss(segs, labels)
        segloss.backward()
        linear_model.optim.step()
        linear_model.optim.zero_grad()

        if i + 1 == args.train_iter:
            est_labels = segs[-1].argmax(1)

    model_path = f"results/linear_{ind}_b{args.train_size}_idmap-{name}.model"
    torch.save(linear_model.state_dict(), model_path)
    
    est_labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in est_labels]
    res = []
    for img, lbl, pred in zip(images, labels_viz, est_labels_viz):
        res.extend([img, lbl, pred])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest") for r in res]
    fpath = f"results/svm_train_{ind}_b{args.train_size}_idmap-{name}.png"
    vutils.save_image(torch.cat(res), fpath, nrow=3)

    global_dic, class_dic, images = test(
        generator, linear_model,
        test_latents, test_noises, args.test_size)
    images = [F.interpolate(img, size=256, mode="nearest") for img in images]

    fpath = fpath.replace("train", "result")
    vutils.save_image(torch.cat(images), fpath, nrow=3)
    np.save(fpath.replace(".png", "global.npy"), global_dic)
    np.save(fpath.replace(".png", "class.npy"), class_dic)

    feats = []
    labels = []
    images = []
    stages = []
    if ind > args.train_size * 4:
        break
