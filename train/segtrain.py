"""
Semantic enhanced discriminator training.
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

cfg = config.SDConfig()
cfg.parse()
cfg.print_info()
cfg.setup()
upsample = int(np.log2(cfg.imsize // 4))

# A little weird because this is not paired with generator
disc = model.simple.Discriminator(upsample=upsample, in_dim=cfg.n_class)
if len(cfg.disc_load_path) > 0:
    state_dict = torch.load(cfg.disc_load_path, map_location='cpu')
    disc.load_state_dict(state_dict)
    del state_dict
disc = disc.to(cfg.device)
disc.train()

sg = model.simple.Generator(upsample=upsample, out_act="none", out_dim=cfg.n_class)
if len(cfg.gen_load_path) > 0:
    state_dict = torch.load(cfg.gen_load_path, map_location='cpu')
    sg.load_state_dict(state_dict)
    del state_dict
sg = sg.to(cfg.device)
sg.train()

softmax = torch.nn.Softmax2d()
softmax = softmax.cuda()

g_optim = torch.optim.Adam(sg.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
d_optim = torch.optim.Adam(disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.9))

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
for ind, sample in enumerate(pbar):
    ind += 1
    _, real_label_compat = sample
    real_label = utils.onehot(real_label_compat, cfg.n_class).cuda()
    latent.normal_()

    with torch.no_grad():
        fake_label_logit = sg(latent, seg=False, detach=False)
        fake_label = softmax(fake_label_logit)

    # Train disc
    disc_loss, grad_penalty = wgan_gp(disc, real_label, fake_label)
    d_optim.step()

    # Train gen
    sg.zero_grad()
    utils.requires_grad(disc, False)
    latent.normal_()
    # detach for prevent semantic branch's gradient to backbone
    gen_label_logit = sg(latent, seg=False, detach=False)
    gen_label = softmax(gen_label_logit)
    disc_gen = disc(gen_label)
    gen_loss = -disc_gen.mean()
    gen_loss.backward()
    g_optim.step()
    utils.requires_grad(disc, True)

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

        p = next(disc.parameters())
        print("Disc: %f %f" % (p.max(), p.min()))
        p = next(sg.parameters())
        print("Gen: %f %f" % (p.max(), p.min()))

    if ind % cfg.save_iter == 0 or ind == cfg.n_iter:
        print(f"=> Snapshot model {ind}")
        torch.save(sg.state_dict(), cfg.expr_dir + "/gen_iter_%06d.model" % ind)
        torch.save(disc.state_dict(), cfg.expr_dir + "/disc_iter_%06d.model" % ind)

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        real_label_viz = []
        num = min(4, real_label.shape[0])
        for i in range(num):
            viz = utils.tensor2label(real_label_compat[i], cfg.n_class)
            real_label_viz.extend([viz])
        real_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear").cpu() for m in real_label_viz])

        fake_label_viz = []
        fake_label_compat = fake_label_logit.argmax(1)
        for i in range(num):
            viz = utils.tensor2label(fake_label_compat[i], cfg.n_class)
            fake_label_viz.extend([viz])
        fake_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear").cpu() for m in fake_label_viz])

        vutils.save_image(real_label_viz, cfg.expr_dir + "/real_label_viz_%05d.png" % ind, nrow=2)
        vutils.save_image(fake_label_viz, cfg.expr_dir + "/fake_label_viz_%05d.png" % ind, nrow=2)
        
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")