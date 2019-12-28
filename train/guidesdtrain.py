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

cfg = config.GuideConfig()
cfg.parse()
cfg.print_info()
cfg.setup()
upsample = int(np.log2(cfg.imsize // 4))

# A little weird because this is not paired with generator
disc = model.simple.Discriminator(upsample=upsample)
if len(cfg.disc_load_path) > 0:
    state_dict = torch.load(cfg.disc_load_path, map_location='cpu')
    disc.load_state_dict(state_dict)
    del state_dict
if cfg.seg > 0:
    disc.set_semantic_config(cfg.disc_semantic_config)
disc = disc.to(cfg.device)
disc.train()

sg = model.simple.Generator(upsample=upsample, semantic=cfg.semantic_config)
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


def wgan(D, x_real, x_fake):
    D.zero_grad()
    disc_real = D(x_real)
    disc_real_loss = - disc_real.mean()
    disc_real_loss.backward()
    disc_fake = D(x_fake)
    disc_fake_loss = disc_fake.mean()
    disc_fake_loss.backward()
    disc_loss = disc_fake_loss + disc_real_loss
    
    return disc_loss

def norm_gradient_penalty(y, x):
    grad = torch.autograd.grad(
        outputs=y, 
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    loss = 10 * ((norm - 1) ** 2).mean()
    return loss

def target_gradient_penalty(y, x, target):
    grad = torch.autograd.grad(
        outputs=y, 
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    loss = 10 * ((grad - target) ** 2).mean()
    return loss

def delta_gradient_guide(D, real_image, fake_image, real_label, fake_label):
    std_image = real_image.view(real_image.size(0), -1).std(1).view(-1, 1, 1, 1)
    std_label = real_label.view(real_label.size(0), -1).std(1).view(-1, 1, 1, 1)
    eps = torch.randn_like(real_image) * std_image
    mid_image = eps + real_image.data
    eps = torch.randn_like(real_label) * std_label
    mid_label = eps + real_label.data
    mid_label = (mid_label - mid_label.min()) / (mid_label.max() - mid_label.min())
    mid_label = mid_label.detach()
    mid_image.requires_grad = True
    mid_label.requires_grad = True

    disc_image = D(torch.cat([mid_image, real_label], 1))
    gp1 = target_gradient_penalty(disc_image.sum(), mid_image, real_image - mid_image)
    gp1.backward()

    disc_label = D(torch.cat([real_image, mid_label], 1))
    gp2 = target_gradient_penalty(disc_label.sum(), mid_label, real_label - mid_label)
    gp2.backward()

    return gp1, gp2

def norm_gradient_guide(D, real_image, fake_image, real_label, fake_label):
    eps.uniform_()
    mid_image = eps * real_image.data + (1 - eps) * fake_image.data
    eps.uniform_()
    mid_label = eps * real_label.data + (1 - eps) * fake_label.data
    mid_image.requires_grad = True
    mid_label.requires_grad = True

    disc_image = D(torch.cat([mid_image, real_label], 1))
    gp1 = norm_gradient_penalty(disc_image.sum(), mid_image)
    gp1.backward()

    disc_label = D(torch.cat([real_image, mid_label], 1))
    gp2 = norm_gradient_penalty(disc_label.sum(), mid_label)
    gp2.backward()

    return gp1, gp2

record = cfg.record

pbar = tqdm(utils.infinite_dataloader(cfg.dl, cfg.n_iter + 1), total=cfg.n_iter)
for ind, sample in enumerate(pbar):
    ind += 1
    real_image, real_label_compat = sample
    real_image = real_image.cuda()
    real_label = utils.onehot(real_label_compat, cfg.n_class).cuda()
    real_label = F.interpolate(real_label, real_image.size(2), mode="bilinear")
    latent.normal_()

    with torch.no_grad():
        fake_image, fake_label_logit = sg(latent)
        fake_label_logit = sg.extract_segmentation(sg.stage)[0]
        fake_label = softmax(fake_label_logit)
    disc_fake_in = torch.cat([fake_image, fake_label], 1) if cfg.seg > 0 else fake_image
    disc_real_in = torch.cat([real_image, real_label], 1) if cfg.seg > 0 else real_image

    # Train disc
    disc_loss = wgan(disc, disc_real_in, disc_fake_in)
    if cfg.guide == "delta":
        gp_image, gp_label = delta_gradient_guide(
            disc, real_image, fake_image, real_label, fake_label)
    elif cfg.guide == "norm":
        gp_image, gp_label = norm_gradient_guide(
            disc, real_image, fake_image, real_label, fake_label)
    d_optim.step()

    # Train gen
    sg.zero_grad()
    utils.requires_grad(disc, False)
    latent.normal_()
    # detach for prevent semantic branch's gradient to backbone
    gen, gen_label_logit = sg(latent, detach=True)
    gen_label = softmax(gen_label_logit)
    disc_gen_in = torch.cat([gen, gen_label], 1) if cfg.seg > 0 else gen
    disc_gen = disc(disc_gen_in)
    gen_loss = -disc_gen.mean()
    gen_loss.backward()
    g_optim.step()
    utils.requires_grad(disc, True)

    # display
    record['disc_loss'].append(utils.torch2numpy(disc_loss))
    record['gp_image'].append(utils.torch2numpy(gp_image))
    record['gp_label'].append(utils.torch2numpy(gp_label))
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
        if cfg.seg > 0:
            real_label_viz = []
            num = min(4, real_label.shape[0])
            for i in range(num):
                img = (real_image[i] + 1) / 2
                viz = utils.tensor2label(real_label_compat[i], cfg.n_class)
                real_label_viz.extend([img, viz])
            real_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear").cpu() for m in real_label_viz])

            fake_label_viz = []
            fake_label_compat = fake_label_logit.argmax(1)
            for i in range(num):
                img = (fake_image[i] + 1) / 2
                viz = utils.tensor2label(fake_label_compat[i], cfg.n_class)
                fake_label_viz.extend([img, viz])
            fake_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear").cpu() for m in fake_label_viz])

            vutils.save_image(real_label_viz, cfg.expr_dir + "/real_label_viz_%05d.png" % ind, nrow=2)
            vutils.save_image(fake_label_viz, cfg.expr_dir + "/fake_label_viz_%05d.png" % ind, nrow=2)
        else:
            vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
            vutils.save_image(real_image[:4], cfg.expr_dir + '/real_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, cfg.expr_dir + "/loss.png")