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
from loss import *
import model

cfg = config.SDConfig()
cfg.parse()
cfg.print_info()
cfg.setup()
step = 8
alpha = 1

# A little weird because this is not paired with generator
state_dict = torch.load(cfg.disc_load_path, map_location='cpu')
disc = model.mixdisc.Discriminator(from_rgb_activate=True)
disc.load_state_dict(state_dict)
disc.set_semantic_config(cfg.disc_semantic_config)
disc = torch.nn.DataParallel(disc)
disc = disc.to(cfg.device)
disc.train()

state_dict = torch.load(cfg.load_path, map_location='cpu')
sg = model.tfseg.StyledGenerator(semantic=cfg.semantic_config)
sg.load_state_dict(state_dict, strict=False)
sg = torch.nn.DataParallel(sg)
sg = sg.to(cfg.device)
sg.train()
del state_dict

g_optim = torch.optim.Adam(sg.module.g_synthesis.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
d_optim = torch.optim.Adam(disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.99))

latent = torch.randn(cfg.batch_size, 512).cuda()
eps = torch.rand(cfg.batch_size, 1, 1, 1).cuda()

bce_crit = torch.nn.BCEWithLogitsLoss()
ones = None
zeros = None
gen_loss_val = 0

record = cfg.record

pbar = tqdm(utils.infinite_dataloader(cfg.dl, cfg.n_iter + 1), total=cfg.n_iter)
for ind, sample in enumerate(pbar):
    ind += 1
    real_image, real_label_compat = sample
    real_image = real_image.cuda()
    real_label = utils.onehot(real_label_compat, cfg.n_class).cuda()
    real_label = F.interpolate(real_label, real_image.size(2), mode="bicubic")
    latent.normal_()
    eps.uniform_()

    with torch.no_grad():
        fake_image, fake_label_logit = sg(latent)
        #fake_label_logit = sg.extract_segmentation()[-1]
        #fake_label = utils.onehot_logit(fake_label_logit, cfg.n_class)
        fake_label = F.softmax(fake_label_logit, 1)
        fake_label = F.interpolate(fake_label, fake_image.size(2), mode="bicubic")
    disc_fake_in = torch.cat([fake_image, fake_label], 1)
    disc_real_in = torch.cat([real_image, real_label], 1)

    # Train disc
    disc.zero_grad()
    # Real
    disc_real, disc_low_real = disc(disc_real_in, step=step, alpha=alpha)
    disc_real_loss = - disc_real.mean()
    if ones is None:
        ones = torch.ones_like(disc_low_real)
    disc_low_real_loss = bce_crit(disc_low_real, ones)
    (disc_real_loss + disc_low_real_loss).backward()
    # Fake
    disc_fake, disc_low_fake = disc(disc_fake_in, step=step, alpha=alpha)
    disc_fake_loss = disc_fake.mean()
    if zeros is None:
        zeros = torch.zeros_like(disc_low_fake)
    disc_low_fake_loss = bce_crit(disc_low_fake, zeros)
    (disc_fake_loss + disc_low_fake_loss).backward()
    disc_loss = disc_fake_loss + disc_real_loss
    disc_low_loss = disc_low_fake_loss + disc_low_real_loss

    # not sure the segmenation mask can be interpolated
    # x_hat = eps * real_image.data + (1 - eps) * fake_image.data
    # y_hat = eps * real_label.data + (1 - eps) * fake_label.data
    x_hat = 0.1 * eps + real_image.data
    y_hat = real_label
    in_hat = torch.cat([x_hat, y_hat], 1)
    in_hat.requires_grad = True
    disc_hat, _ = disc(in_hat, step=step, alpha=alpha)
    grad_in_hat = torch.autograd.grad(
        outputs=disc_hat.sum(), 
        inputs=in_hat,
        create_graph=True)[0]
    grad_in_hat_norm = grad_in_hat.view(grad_in_hat.size(0), -1).norm(2, dim=1)
    grad_penalty = 10 * ((grad_in_hat_norm - 1) ** 2).mean()
    grad_penalty.backward()
    d_optim.step()

    # Train gen
    if ind % cfg.n_critic == 0:
        sg.zero_grad()
        utils.requires_grad(disc, False)
        latent.normal_()
        gen, gen_label_logit = sg(latent)
        #this is not differentiable
        #gen_label = utils.onehot_logit(gen_label_logit, cfg.n_class)
        gen_label = F.softmax(gen_label_logit, 1)
        gen_label = F.interpolate(gen_label, gen.size(2), mode="bicubic")
        disc_gen_in = torch.cat([gen, gen_label], 1)
        disc_gen, disc_low_gen = disc(disc_gen_in, step=step, alpha=alpha)
        gen_loss = -disc_gen.mean()
        disc_low_gen_loss = bce_crit(disc_low_gen, ones)
        (gen_loss + disc_low_gen_loss).backward()
        utils.requires_grad(disc, True)
        gen_loss_val = gen_loss.item()

    # display
    record['disc_loss'].append(utils.torch2numpy(disc_loss))
    record['disc_low_loss'].append(utils.torch2numpy(disc_low_loss))
    record['grad_penalty'].append(utils.torch2numpy(grad_penalty))
    record['gen_loss'].append(gen_loss_val)

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

    if ind % cfg.save_iter == 0 or ind == cfg.n_iter:
        print(f"=> Snapshot model {ind}")
        torch.save(sg.state_dict(), cfg.expr_dir + "/gen_iter_%06d.model" % ind)
        torch.save(disc.state_dict(), cfg.expr_dir + "/disc_iter_%06d.model" % ind)

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                            nrow=2, normalize=True, range=(-1, 1))

        real_label_viz = []
        for i in range(real_label.shape[0]):
            viz = utils.tensor2label(real_label_compat[i], cfg.n_class)
            real_label_viz.append(F.interpolate(viz.unsqueeze(0), 256))
        real_label_viz = torch.cat(real_label_viz)

        gen_label_viz = []
        gen_label_compat = gen_label.argmax(1, keepdim=True)
        for i in range(gen_label.shape[0]):
            viz = utils.tensor2label(gen_label_compat[i], cfg.n_class)
            gen_label_viz.append(F.interpolate(viz.unsqueeze(0), 256))
        gen_label_viz = torch.cat(gen_label_viz)

        vutils.save_image(gen_label_viz, cfg.expr_dir + "/gen_label_viz_%05d.png" % ind, nrow=2)
        vutils.save_image(real_label_viz, cfg.expr_dir + "/real_label_viz_%05d.png" % ind, nrow=2)
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")