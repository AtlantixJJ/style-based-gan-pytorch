import sys, os, argparse
sys.path.insert(0, ".")
import torch, glob
import numpy as np
from lib.face_parsing.unet import unet
from tqdm import tqdm
from torchvision import utils as vutils
import model, fid, utils, evaluate, dataset


imsize = 128
device = "cuda"
model_dir = f"expr/wgan128"
model_files = glob.glob(f"{model_dir}/*.model")
model_files = [m for m in model_files if "disc" not in m]
model_files.sort()

upsample = int(np.log2(imsize // 4))
generator = model.simple.Generator(upsample=upsample)
generator.to(device).eval()

torch.manual_seed(65537)
latents = torch.randn((25, 128), device=device)

for i, model_file in enumerate(tqdm(model_files)):
    state_dict = torch.load(model_file)
    missed = generator.load_state_dict(state_dict)
    
    img = generator(latents).clamp(-1, 1)
    vutils.save_image((img+1)/2, f"{model_dir}/fix_{i}.png", nrow=5)
