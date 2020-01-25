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
parser.add_argument("--reg", default=0, type=int)
args = parser.parse_args()

def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))

# constants setup
INTERVAL = 20
TOTAL = INTERVAL * 7
torch.manual_seed(1)
device = 'cuda'
LR = 0.2
# input
latent = torch.randn(1, 512).to(device)
latent.requires_grad = True
noises = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noises.append(torch.randn(1, 1, size, size).cuda())

# set up mask
mask = torch.zeros(1, 1, 1024, 1024, dtype=torch.float32).to(device)
mask[0, 0, 280:340, 250:310] = 1

# build model
generator = 0
if args.reg == 1:
    generator = model.tfseg.StyledGenerator(semantic=args.seg_cfg).to(device)
elif args.reg == 2:
    generator = model.tfseg_reg.StyledGenerator(semantic=args.seg_cfg).to(device)
generator.load_state_dict(torch.load(args.model))
generator.eval()
utils.requires_grad(generator, False)
generator.set_noise(noises)
generator.semantic_branch.get_orthogonal_weight()

with torch.no_grad():
    orig_image, orig_seg = generator(latent)
orig_stage = [s.clone() for s in generator.stage]
orig_image = (orig_image.clamp(-1, 1) + 1) / 2
orig_label = orig_seg.argmax(1).long()
revise_bias = [torch.zeros_like(t) for t in orig_stage if t.shape[3] >= 16]
for r in revise_bias:
    r.requires_grad = True

optim = torch.optim.SGD([latent], lr=1e-2)
optim_revision = torch.optim.SGD(revise_bias, lr=1e-2)

# target
y = torch.zeros(1, 3, 1024, 1024).cuda()
y.fill_(-1)
y[0, 2] = 0.7

logsoftmax = torch.nn.CrossEntropyLoss()
logsoftmax = logsoftmax.cuda()

record = {"mseloss": [], "celoss": [], "segdiff": []}

res = []
for i in tqdm(range(TOTAL)):
    image, seg = generator(latent)
    new_label = seg.argmax(1)

    mseloss = ((image - y) * mask) ** 2
    mseloss = mseloss.sum() / mask.sum()
    latent.grad = torch.autograd.grad(
        outputs=mseloss,
        inputs=latent,
        only_inputs=True)[0]
    optim.step()

    count = 0
    total_diff = 0
    while count < 10:
        if args.reg == 1:
            image, seg = generator(latent)
        elif args.reg == 2:
            image, seg = generator(latent, bias=revise_bias)
        mseloss = ((image - y) * mask) ** 2
        mseloss = mseloss.sum() / mask.sum()
        revise_label = seg.argmax(1).long()
        celoss = logsoftmax(seg, orig_label)
        # directly use cross entropy may also decrease other part
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        count += 1
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)
        if total_diff < 100:
            break
        grad_seg = torch.autograd.grad(
                outputs=celoss,
                inputs=seg,
                only_inputs=True)[0]
        if args.reg == 1:
            latent.grad = torch.autograd.grad(
                outputs=seg,
                inputs=latent,
                grad_outputs=grad_seg * diff_mask,
                only_inputs=True)[0]
            optim.step()
        elif args.reg == 2:
            bias_grads = torch.autograd.grad(
                outputs=seg,
                inputs=revise_bias,
                grad_outputs=grad_seg * diff_mask,
                only_inputs=True)
            for r,g in zip(revise_bias, bias_grads):
                r.grad = g
            optim_revision.step()

    if (i + 1) % INTERVAL == 0:
        res.append((image.clamp(-1, 1) + 1) / 2)
        res.append(utils.tensor2label(new_label, 16).unsqueeze(0))

res.append(orig_image * (1 - mask) + y * mask)
res.append(utils.tensor2label(orig_label, 16).unsqueeze(0))
res = torch.cat([r.cpu() for r in res])

vutils.save_image(res, f"celossreg_edit_{args.reg}_sample.png", nrow=4)
utils.plot_dic(record, f"celossreg_edit_{args.reg}_loss.png")
