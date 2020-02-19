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
import evaluate, utils, dataset
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
    "--layer-index", default="4", type=str)
parser.add_argument(
    "--train-size", default=4, type=int)
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

# setup test
test_latents = torch.randn(args.test_size, 512)
test_noises = [generator.generate_noise() for _ in range(args.test_size)]

# set up input
latent = torch.randn(1, 512).to(device)
layer_index = [int(i) for i in args.layer_index.split(",")]
colorizer = utils.Colorize(args.total_class)
stylegan_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
stylegan_dims = [stylegan_dims[l] for l in layer_index]
linear_model = 0

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
    feat = torch.cat([F.interpolate(f, size=maxsize, mode="bilinear") for f in feat], 1)
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
        est_label = linear_model.predict(stage)
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
    return global_dic, class_dic, result


images = []
stages = []
feats = []
labels = []
for ind, sample in enumerate(tqdm(dl)):
    latent, noise, image, label = sample
    latent = latent[0].to(device)
    label = label[:, :, :, 0].unsqueeze(0)
    label = idmap(label)
    labels.append(label)

    image = generator(latent, seg=False)
    stages.append([generator.stage[l].detach() for l in layer_index])
    feat = get_feature(generator, latent, noise, layer_index)
    feats.append(feat)
    image = image.clamp(-1, 1).detach().cpu()
    images.append((image + 1) / 2)

    if (ind + 1) % args.train_size != 0:
        continue

    feats = torch.cat(feats)
    labels = torch.cat(labels)
    labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in labels]

    print(f"=> Feature shape: {feats.shape}")
    print(f"=> Label shape: {labels.shape}")
    N, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C).float().cpu().numpy()
    print(f"=> Feature for SVM shape: {feats.shape}")
    labels = F.interpolate(labels.float(), size=H, mode="nearest").long().numpy()

    if args.ovo:
        svm_model = SVM(kernel="linear")
        svm_model.fit(feats, labels.reshape(-1))
        model_path = f"results/svm_ovo_{ind}_l{args.layer_index}_b{args.train_size}_idmap-{name}.model"
        svm_model.save_to_file(model_path)
        est_labels = svm_model.predict(feats)

        linear_model = OVOLinearSemanticExtractor(
            len(contiguous_label),
            stylegan_dims,
            inv_idmap).to(device)
        linear_model.copy_weight_from(svm_model.coef_, svm_model.intercept_)
    else:
        svm_model = svm.train(labels.reshape(-1), feats, "-n 8 -s 2 -B -1 -q")
        model_path = f"results/svm_ovr_{ind}_l{args.layer_index}_b{args.train_size}_idmap-{name}.model"
        svm.save_model(model_path, svm_model)

        n_class = svm_model.get_nr_class()
        map_from = list(range(n_class))
        map_to = svm_model.get_labels()
        id2cid = {fr:to for fr, to in zip(map_from, map_to)}
        inv_idmap = lambda x: utils.idmap(x, id2cid=id2cid)
        coef = np.array([svm_model.get_decfun(i)[0] for i in range(n_class)])
        linear_model = LinearSemanticExtractor(
            n_class,
            stylegan_dims,
            inv_idmap).to(device)
        linear_model.copy_weight_from(coef)
        est_labels = np.concatenate([linear_model.predict(stage) for stage in stages])
    
    est_labels = torch.from_numpy(est_labels)
    est_labels_viz = [colorizer(l).unsqueeze(0).float() / 255. for l in est_labels]
    res = []
    for img, lbl, pred in zip(images, labels_viz, est_labels_viz):
        res.extend([img, lbl, pred])
    res = [F.interpolate(r.detach().cpu(), size=256, mode="nearest") for r in res]
    fpath = f"results/svm_train_{ind}_l{args.layer_index}_b{args.train_size}_idmap-{name}.png"
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
