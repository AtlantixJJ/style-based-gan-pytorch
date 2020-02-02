import sys
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
import numpy as np
import model
import utils

device = "cpu"
generator = model.tfseg_fuse.StyledGenerator()
state_dict = torch.load(sys.argv[1], map_location=device)
missed = generator.load_state_dict(state_dict, strict=False)
print(missed)
generator = generator.to(device)
torch.manual_seed(65539)
x1 = torch.randn(1, 512).to(device)
x2 = torch.randn(1, 512).to(device)

colorizer = utils.Colorize(16)
image1, label1 = generator.calc(x1)
image2, label2 = generator.calc(x2)
mask = (label1 == 10).float().unsqueeze(0) # 1 for skin
image, seg = generator.spatial_mix_forward(x1, x2, mask)
label = seg.argmax(1).cpu()
image = (image.cpu().clamp(-1, 1) + 1) / 2
label_viz  = colorizer(label ).unsqueeze(0) / 255.
label1_viz = colorizer(label1).unsqueeze(0) / 255.
label2_viz = colorizer(label2).unsqueeze(0) / 255.
images = [
    image1, label1_viz,
    image,  label_viz,
    image2, label2_viz]
images = torch.cat(images)
vutils.save_image(images, "spatial_mix_forward.png", nrow=2)