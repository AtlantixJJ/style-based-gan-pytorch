import sys
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
import numpy as np
import model

state_dict = torch.load(sys.argv[1])
resolution = 0
if "256x256" in sys.argv[1]:
    resolution = 256
elif "1024x1024" in sys.argv[1]:
    resolution = 1024
generator = model.tf.StyledGenerator(resolution)
missed = generator.load_state_dict(state_dict, strict=False)
print(missed)
generator = generator.cuda()
x = torch.randn(4, 512).cuda()
y = generator(x)
y = (y.clamp(-1, 1) + 1) / 2
vutils.save_image(y, "gen.png", nrow=2)