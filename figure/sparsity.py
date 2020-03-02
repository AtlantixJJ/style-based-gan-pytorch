import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model
from model.semantic_extractor import get_semantic_extractor


device = 'cpu'
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
generator = model.load_stylegan(model_path).to(device)
latent = torch.randn(1, 512).to(device)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]

data_dir = "record/l1"
model_files = glob.glob(f"{data_dir}/*")
model_files = [f for f in model_files if ".npy" not in f]
model_files.sort()
model_files = [f"{m}/stylegan_linear_extractor.model"
    for m in model_files]
func = get_semantic_extractor("linear")

def concat_weight(module):
    vals = []
    ws = []
    for i, conv in enumerate(module):
        w = conv[0].weight[:, :, 0, 0]
        ws.append(w)
    ws = torch.cat(ws, 1)
    return ws

def calc_subset(dic):
    indice = {}
    indice["face"] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]
    # eye glasses, ear ring, neck lace, hat, cloth
    indice["other"] = [3, 11, 12, 13, 15]
    for metric in ["IoU"]:
        for t in ["face", "other"]:
            arr = np.array(dic[metric])
            v = arr[indice[t]]
            v = v[v>-1]
            dic[f"m{metric}_{t}"] = v.mean()
    return dic

s = []
for model_file in model_files:
    sep_model = func(n_class=16, dims=dims)
    sep_model.load_state_dict(torch.load(model_file))
    w = concat_weight(sep_model.semantic_extractor)
    sparsity = w.abs().sum() / w.shape[0]
    file_name = model_file.replace(
        "/stylegan_linear_extractor.model",
        "_agreement.npy")
    dic = np.load(file_name, allow_pickle=True)[()]
    mIoU = dic['mIoU']
    dic["IoU"][13] = -1
    arr = np.array(dic["IoU"])
    mIoU = arr[arr>-1].mean()
    dic = calc_subset(dic)
    mIoU_face = dic['mIoU_face']
    mIoU_other = dic['mIoU_other']
    s.append("%f,%f,%f,%f" % (sparsity, mIoU, mIoU_face, mIoU_other))
with open("sparsity.csv", "w") as f:
    f.write("\n".join(s))