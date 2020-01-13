"""
Use cross entropy loss to regularize the edit
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
import numpy as np
from PIL import Image
import model
import utils
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="expr/fixseg_1.0_mul-16/iter_010000.model")
parser.add_argument("--seg-cfg", default="mul-16")
args = parser.parse_args()

def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))

# constants setup
torch.manual_seed(1)
device = 'cuda'
LR = 0.2
# input
latent = torch.randn(1, 512).to(device)
noises = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noises.append(torch.randn(1, 1, size, size).cuda())

# set up mask
mask = torch.zeros(1, 1, 1024, 1024, dtype=torch.float32).to(device)
mask[0, 0, 280:340, 250:310] = 1

# build model
generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()
utils.requires_grad(generator, False)
generator.set_noise(noises)
generator.semantic_branch.get_orthogonal_weight()

with torch.no_grad():
    extended_latent = generator.g_mapping(latent)
    orig_image, orig_stage = generator.g_synthesis(extended_latent)
    seg = generator.extract_segmentation(orig_stage)[-1]

orig_image = (orig_image.clamp(-1, 1) + 1) / 2
orig_label = seg.argmax(1).long()
extended_latent = extended_latent.clone()
extended_latent.requires_grad = True
optim = torch.optim.SGD([extended_latent], lr=1e-2)

# target
y = torch.zeros(1, 3, 1024, 1024).cuda()
y.fill_(-1)
y[0, 2] = 1.0 # RGB = (0, 255, 0)

logsoftmax = torch.nn.CrossEntropyLoss()
logsoftmax = logsoftmax.cuda()

record = {"mseloss": [], "celoss": [], "segdiff": []}

res = []
for i in tqdm(range(71)):
    image, stage = generator.g_synthesis(extended_latent)
    seg = generator.extract_segmentation(stage)[-1]
    new_label = seg.argmax(1)

    mseloss = ((image - y) * mask) ** 2
    mseloss = mseloss.sum() / mask.sum()
    extended_latent.grad = torch.autograd.grad(
        outputs=mseloss,
        inputs=extended_latent,
        only_inputs=True)[0]
    optim.step()

    count = 0
    total_diff = 0
    while count < 10:
        image, stage = generator.g_synthesis(extended_latent)
        seg = generator.extract_segmentation(stage)[-1]
        revise_label = seg.argmax(1).long()
        # directly use cross entropy may also decrease other part
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        if total_diff < 100:
            break
        celoss = logsoftmax(seg, orig_label)
        grad_seg = torch.autograd.grad(
            outputs=celoss,
            inputs=seg,
            only_inputs=True)[0]
        extended_latent.grad = torch.autograd.grad(
            outputs=seg,
            inputs=extended_latent,
            grad_outputs=grad_seg * diff_mask,
            only_inputs=True)[0]
        optim.step()
        count += 1
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)

    if (i + 1) % 10 == 0:
        res.append((image.clamp(-1, 1) + 1) / 2)
        res.append(utils.tensor2label(new_label, 16).unsqueeze(0))

res.append(orig_image * (1 - mask) + y * mask)
res.append(utils.tensor2label(orig_label, 16).unsqueeze(0))
res = torch.cat([r.cpu() for r in res])

vutils.save_image(res, f'ortho_reg_edit_sample.png', nrow=4)
utils.plot_dic(record, "ortho_reg_edit_loss.png")
