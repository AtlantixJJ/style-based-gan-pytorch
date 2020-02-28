"""
Semantic enhanced discriminator training.
python script/wgan.py --imsize 64
# hat128
python train/wgan.py --imsize 128 --load expr/celeba_wgan128/gen_iter_050000.model --disc-net expr/celeba_wgan128/disc_iter_050000.model --batch-size 64 --gpu 0 --save-iter 10 --disp-iter 10 --iter-num 500 --dataset ../datasets/CelebAMask-HQ/hat/image --task celeba_hat_wgan --lr 0.0002 --warmup 100
# eyeg128
python train/wgan.py --imsize 128 --load expr/celeba_wgan128/gen_iter_050000.model --disc-net expr/celeba_wgan128/disc_iter_050000.model --batch-size 64 --gpu 0 --save-iter 10 --disp-iter 10 --iter-num 500 --dataset ../datasets/CelebAMask-HQ/eye_g/image --task celeba_eyeg_wgan --lr 0.0002 --warmup 100
"""
import sys
sys.path.insert(0, ".")
import os
from IPython.core.debugger import set_trace
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from lib.face_parsing import unet
import config
import utils
import model

cfg = config.BasicGANConfig()
cfg.parse()
s = str(cfg)
cfg.setup()
print(s)
with open(cfg.expr_dir + "/config.txt", "w") as f:
	f.write(s)
upsample = int(np.log2(cfg.imsize // 4))

# A little weird because this is not paired with generator
disc = model.simple.Discriminator(upsample=upsample)
if len(cfg.disc_load_path) > 0:
    state_dict = torch.load(cfg.disc_load_path, map_location='cpu')
    disc.load_state_dict(state_dict)
    del state_dict
disc = torch.nn.DataParallel(disc)
disc = disc.to(cfg.device)
disc.train()

sg = model.simple.Generator(upsample=upsample)
if len(cfg.gen_load_path) > 0:
    state_dict = torch.load(cfg.gen_load_path, map_location='cpu')
    sg.load_state_dict(state_dict)
    del state_dict
sg = torch.nn.DataParallel(sg)
sg = sg.to(cfg.device)
sg.train()

g_optim = torch.optim.Adam(sg.module.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
d_optim = torch.optim.Adam(disc.module.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.9))


g_sched = torch.optim.lr_scheduler.LambdaLR(
    g_optim,
    lambda step: min((float(step) + 1) / (cfg.warmup + 1), 1))
d_sched = torch.optim.lr_scheduler.LambdaLR(
    d_optim,
    lambda step: min((float(step) + 1) / (cfg.warmup + 1), 1))

latent = torch.randn(cfg.batch_size, 128).cuda()
eps = torch.rand(cfg.batch_size, 1, 1, 1).cuda()


def wgan_gp(D, x_real, x_fake):
    eps.uniform_()
    D.zero_grad()
    disc_real = D(x_real)
    disc_real_loss = - disc_real.mean()
    disc_real_loss.backward()
    disc_fake = D(x_fake)
    disc_fake_loss = disc_fake.mean()
    disc_fake_loss.backward()
    disc_loss = disc_fake_loss + disc_real_loss

    # not sure the segmenation mask can be interpolated
    x_hat = eps * x_real.data + (1 - eps) * x_fake.data
    # DRAGAN
    # std_x = x_real.view(x_real.size(0), -1).std(1).view(-1, 1, 1, 1)
    # x_hat = std_x * eps + (1 - eps) * x_real.data
    x_hat.requires_grad = True
    disc_hat = D(x_hat)
    grad_x_hat = torch.autograd.grad(
        outputs=disc_hat.sum(), 
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    grad_x_hat_norm = grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1)
    grad_penalty = 10 * ((grad_x_hat_norm - 1) ** 2).mean()
    grad_penalty.backward()

    return disc_loss, grad_penalty


record = cfg.record

pbar = tqdm(utils.infinite_dataloader(cfg.dl, cfg.n_iter + 1), total=cfg.n_iter)
for ind, real_image in enumerate(pbar):
    ind += 1
    real_image = real_image.cuda()
    latent.normal_()

    fake_image = sg(latent).detach()

    # Train disc
    disc_loss, grad_penalty = wgan_gp(disc, real_image, fake_image)
    d_optim.step()

    flag = (disc_loss + grad_penalty > 500)

    # Train gen
    sg.zero_grad()
    utils.requires_grad(disc, False)
    latent.normal_()
    gen = sg(latent)
    gen_loss = -disc(gen).mean()
    gen_loss.backward()
    g_optim.step()
    utils.requires_grad(disc, True)

    # schedule lr for warm start
    g_sched.step()
    d_sched.step()

    # display
    record['disc_loss'].append(utils.torch2numpy(disc_loss))
    record['grad_penalty'].append(utils.torch2numpy(grad_penalty))
    record['gen_loss'].append(utils.torch2numpy(gen_loss))

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

        p = next(disc.module.parameters())
        print("Disc: %f %f" % (p.max(), p.min()))
        p = next(sg.module.parameters())
        print("Gen: %f %f" % (p.max(), p.min()))

    if ind % cfg.save_iter == 0 or ind == cfg.n_iter:
        print(f"=> Snapshot model {ind}")
        #torch.save(g_optim.state_dict(), cfg.expr_dir + "/gen_iter_%06d.state" % ind)
        #torch.save(d_optim.state_dict(), cfg.expr_dir + "/disc_iter_%06d.state" % ind)
        torch.save(sg.module.state_dict(), cfg.expr_dir + "/gen_iter_%06d.model" % ind)
        torch.save(disc.module.state_dict(), cfg.expr_dir + "/disc_iter_%06d.model" % ind)

    if flag:
        print(f"!> Abnormal loss at {ind}")
        vutils.save_image(gen[:16], cfg.expr_dir + '/gen_%06d.png' % ind,
                            nrow=4, normalize=True, range=(-1, 1))
        vutils.save_image(real_image[:16], cfg.expr_dir + '/real_%06d.png' % ind,
                            nrow=4, normalize=True, range=(-1, 1))

    if flag or ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                            nrow=2, normalize=True, range=(-1, 1))
        vutils.save_image(real_image[:4], cfg.expr_dir + '/real_%06d.png' % ind,
                            nrow=2, normalize=True, range=(-1, 1))
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")