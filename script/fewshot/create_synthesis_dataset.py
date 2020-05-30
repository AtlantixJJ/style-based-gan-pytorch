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
    "--output", default="datasets/Synthesized")
parser.add_argument(
    "--number", default=1000, type=int)
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

for folder in ["latent", "noise", "label", "image"]:
    os.system(f"mkdir {args.output}/{folder}")

for ind in tqdm(range(args.number)):
    latent_path = f"{args.output}/latent/{ind:05d}.npy"
    noise_path = f"{args.output}/noise/{ind:05d}.npy"
    label_path = f"{args.output}/label/{ind:05d}.png"
    image_path = f"{args.output}/image/{ind:05d}.png"
    latent.normal_()
    with torch.no_grad():
        image = generator(latent)
        image, stage = generator.get_stage(latent)
        image = image.clamp(-1, 1)
        noise = generator.get_noise()
    label = external_model.segment_batch(image)
    #label = sep_model(stage)[0].argmax(1)

    utils.imwrite(label_path, utils.torch2numpy(label[0]))
    vutils.save_image((image + 1) / 2, image_path)
    np.save(latent_path, utils.torch2numpy(latent).astype("float32"))
    np.save(noise_path, utils.torch2numpy(noise).astype("float32"))