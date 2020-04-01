import numpy as np
import sys
sys.path.insert(0, ".")
import pickle, argparse, glob
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

import model, utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="/home/jianjin/large/multiple_kmeans")
parser.add_argument("--resume", default="")
args = parser.parse_args()

device = "cpu"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)
files = glob.glob(f"{args.dataset}/skm*.pkl")
files.sort()
seed = int(open(f"{args.dataset}/kmeans_feats_seed", "r").read())

torch.manual_seed(seed)
latent_size = 512
maxsize = 512
N = 16
latent = torch.randn(N, latent_size).to(device)

for i, f in enumerate(files):
    name = f[f.rfind("/") + 1:-4]
    print(name)
    centeroids, avg_distance = pickle.load(open(f, "rb")) # (M, C)
    print(centeroids.shape, avg_distance)

    images = []
    for j in range(8):
        with torch.no_grad():
            image, stage = generator.get_stage(latent[j:j+1])
            image = F.interpolate(image, size=512, mode="bilinear")
            images.append((image.clamp(-1, 1) + 1) / 2)
            size = max([s.shape[2] for s in stage])
            data = torch.cat([
                F.interpolate(s.cpu(), size=maxsize, mode="bilinear")[0]
                    for s in stage]) # (C, H, W)
            C, H, W = data.shape
            ind = torch.randperm(data.shape[1] * data.shape[2])
            ind = ind[:len(ind) // N]
            data = utils.torch2numpy(data.view(data.shape[0], -1).permute(1, 0))
        # data: (N, C)
        labels = np.matmul(data, centeroids.transpose()).argmax(1)
        labels = torch.from_numpy(labels.reshape(H, W))
        label_viz = utils.tensor2label(labels, centeroids.shape[0]).unsqueeze(0)
        images.append(label_viz)
    images = torch.cat(images)
    vutils.save_image(images, f"{f[:-4]}_result.png")