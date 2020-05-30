import sys
sys.path.insert(0, ".")
import os
import torch
from torchvision import utils as vutils
from tqdm import tqdm
from PIL import Image
import numpy as np
from model.seg import StyledGenerator
import config
from utils import *
from loss import *

STEP = 8
ALPHA = 1
torch.manual_seed(4)

cfg = config.BaseConfig()
cfg.parse()
cfg.print_info()
cfg.setup()

state_dicts = torch.load(cfg.load_path, map_location='cpu')

sg = StyledGenerator(512)
if "stylegan-1024px-new.model" in cfg.load_path:
	sg.load_state_dict(state_dicts['generator'])
else:
	sg.load_state_dict(state_dicts)
sg.eval()
sg = sg.cuda()

latent = torch.randn(1, 512).cuda()
noise = [0] * (STEP + 1)
for k in range(STEP + 1):
    size = 4 * 2 ** k
    noise[k] = torch.randn(1, 1, size, size).cuda()
images = sg.all_level_forward(latent, step=STEP, alpha=ALPHA, noise=noise)
result = torch.cat([F.interpolate(img, 256, mode='bilinear')
			for img in images])
vutils.save_image(result, "./all_level_forward.png", nrow=4, normalize=True, range=(-1, 1))