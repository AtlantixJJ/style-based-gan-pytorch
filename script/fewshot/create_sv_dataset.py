"""
Calculate the feature of StyleGAN and do RCC clustering.
"""
import sys
sys.path.insert(0, ".")
import os, argparse, pickle
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import utils as vutils

import model, utils
from segmenter import get_segmenter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--number", default=100, type=int)
parser.add_argument(
    "--seed", default=65537, type=int) # 1314 for test
args = parser.parse_args()


device = 'cuda'

extractor_path = "record/vbs_conti/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs8_l10.0001/stylegan_unit_extractor.model"
model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth" if "celebahq" in extractor_path else "checkpoint/face_ffhq_1024x1024_stylegan2.pth"

generator = model.load_model(model_path)
generator.to(device).eval()
torch.manual_seed(args.seed)
latent = torch.randn(1, 512, device=device)
with torch.no_grad():
    image, stage = generator.get_stage(latent)
    image = (image.clamp(-1, 1) + 1) / 2
dims = [s.shape[3] for s in stage]

external_model = get_segmenter(
    "celebahq",
    "checkpoint/faceparse_unet_512.pth")
#sep_model = model.semantic_extractor.get_semantic_extractor("unit")(
#    n_class=15,
#    dims=dims)
#sep_model.load_state_dict(torch.load(args.external_model))

# setup
feats = []
labels = []
for ind in tqdm(range(args.number)):
    latent.normal_()
    with torch.no_grad():
        image = generator(latent)
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        noise = generator.get_noise()
        label = external_model.segment_batch(image, resize=False)[0]
        #label = sep_model(stage)[0].argmax(1)
        stage = stage[3:8] # layers 3~7 is useful
        maxsize = max(s.shape[3] for s in stage)
        feat = torch.cat([utils.bu(s, maxsize)[0] for s in stage])

    # get the (approximated) support vectors
    mask = torch.Tensor(maxsize, maxsize).byte().to(device)
    mask[:-1] = label[:-1] != label[1:] # left - right
    mask[1:] |= mask[:-1] # right - left
    mask[:, :-1] |= label[:, :-1] != label[:, 1:] # top - bottom
    mask[:, 1:] |= mask[:, :-1] # bottom - top
    mask_viz = mask.float().unsqueeze(0).unsqueeze(0)
    
    feats.append(utils.torch2numpy(feat[:, mask].transpose(1, 0)))
    labels.append(utils.torch2numpy(label[mask]))

np.save("sv_feat", np.concatenate(feats))
np.save("sv_label", np.concatenate(labels))
    
