"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, glob
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model.tfseg import StyledGenerator
import argparse
import torch.nn.functional as F
from torchvision import utils as vutils
from lib.face_parsing import unet
import evaluate, utils, dataset
from model.linear import LinearSemanticExtractor, OVOLinearSemanticExtractor
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--svm-model", default="results/svm_model")
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--gpu", default="0")
parser.add_argument(
    "--test-size", default=1000, type=int)
parser.add_argument(
    "--total-class", default=16, type=int)
parser.add_argument(
    "--ovo", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.ovo:
    print("=> Use One v.s. One from thundersvm")
    from thundersvm import SVC as SVM
else:
    print("=> Use One v.s. Rest from liblinear (multicore)")
    import lib.liblinear.liblinearutil as svm

# constants setup
torch.manual_seed(1)
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

test_size = args.test_size
test_latents = torch.randn(test_size, 512)
test_noises = [generator.generate_noise() for _ in range(test_size)]


def get_feature(generator, latent, noise, layer_index):
    feat = [generator.stage[i].detach().cpu() for i in layer_index]
    maxsize = max([f.shape[2] for f in feat])
    feat = torch.cat([F.interpolate(f, size=maxsize, mode="bilinear", align_corners=True) for f in feat], 1)
    return feat.detach()


def test(generator, linear_model, test_latents, test_noises, N):
    result = []
    evaluator = evaluate.MaskCelebAEval()
    for i in tqdm(range(N)):
        latent = test_latents[i:i+1].to(device)
        noise = [n.to(latent.device) for n in test_noises[i]]
        generator.set_noise(noise)
        image, seg = generator(latent)
        stage = [generator.stage[l] for l in layer_index]
        est_label = linear_model.predict(stage, last_only=True)
        size = est_label.shape[2]
        
        seg = F.interpolate(seg, size=size, mode="bilinear", align_corners=True)
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
    return global_dic, class_dic, result


def config_from_name(name):
    ind = name.rfind("/")
    name = name[ind+1:]
    return name.split("_")[3][1:]

model_files = glob.glob(args.svm_model + "/*.model")
model_files.sort()
model_file = model_files[-1]
svm_model = svm.load_model(model_file)

# setup test
test_latents = torch.randn(args.test_size, 512)
test_noises = [generator.generate_noise() for _ in range(args.test_size)]

# set up input
latent = torch.randn(1, 512).to(device)
layer_index = config_from_name(model_file)
layer_index = [int(i) for i in layer_index.split(",")]
colorizer = utils.Colorize(args.total_class)
stylegan_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
stylegan_dims = [stylegan_dims[l] for l in layer_index]
linear_model = 0