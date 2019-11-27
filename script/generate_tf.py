import sys
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
import numpy as np
import model

generator = model.tf.StyledGenerator()
state_dict = torch.load(sys.argv[1])
missed = generator.load_state_dict(state_dict, strict=False)
print(missed)
generator = generator.cuda()
x = torch.randn(4, 512).cuda()
y = generator(x)
y = (y.clamp(-1, 1) + 1) / 2
vutils.save_image(y, "gen.png", nrow=2)
ys = generator.all_layer_forward(x)
for i, y in enumerate(ys):
    y = (y.clamp(-1, 1) + 1) / 2
    vutils.save_image(y, f"gen_{i}.png", nrow=2)