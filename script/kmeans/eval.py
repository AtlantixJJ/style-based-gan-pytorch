import numpy as np
import sys
sys.path.insert(0, ".")
import pickle, argparse, glob
import torch
import torch.nn.functional as f

import model, utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="/home/jianjinxu/large/multiple_kmeans")
parser.add_argument("--resume", default="")
args = parser.parse_args()

device = "cpu"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)
files = glob.glob(f"{args.dataset}/kmeans_feats*.pkl")
files.sort()
seed = int(open(f"{args.dataset}/kmeans_feats_seed", "r").read())

torch.manual_seed(args.seed)
latent_size = 512
maxsize = 512
N = 16
latent = torch.randn(N, latent_size).to(device)
bundle = []
images = []


for i, f in enumerate(files):
    name = f[f.rfind("/") + 1:-3]
    print(name)
    centeroids = pickle.load(open(f, "rb"))
    print(centeroids.shape)

    with torch.no_grad():
        image, stage = generator.get_stage(latent[i:i+1])
        images.append((image.clamp(-1, 1) + 1) / 2)
        size = max([s.shape[2] for s in stage])
        data = torch.cat([
            F.interpolate(s.cpu(), size=maxsize, mode="bilinear")[0]
                for s in stage]) # (C, H, W)
        ind = torch.randperm(data.shape[1] * data.shape[2])
        ind = ind[:len(ind) // N]
        data = utils.torch2numpy(data.view(data.shape[0], -1).permute(1, 0)) # (C, N)
    
    labels = np.matmul(centeroids, data).argmax(0)
    utils.numpy2label(labels, )
