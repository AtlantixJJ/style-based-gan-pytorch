import torch
import torchvision.utils as vutils
import numpy as np
from model.tf import StyledGenerator

generator = StyledGenerator()
state_dict = torch.load("checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt")
generator.load_state_dict(state_dict, strict=False)
generator = generator.cuda()
x = torch.randn(4, 512).cuda()
y = generator(x)
vutils.save_image(y, "gen.png", nrow=2)