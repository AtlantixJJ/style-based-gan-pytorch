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
upsample = int(np.log2(cfg.imsize // 4))

image_disc = model.simple.Discriminator(upsample=upsample, in_dim=3)
image_disc = image_disc.to(cfg.device)
image_disc.train()

seg_disc = model.simple.Discriminator(upsample=upsample, in_dim=cfg.n_class)
seg_disc = seg_disc.to(cfg.device)
seg_disc.train()

sg = model.simple.Generator(upsample=upsample, semantic=cfg.semantic_config)
if len(cfg.gen_load_path) > 0:
    state_dict = torch.load(cfg.gen_load_path, map_location='cpu')
    sg.load_state_dict(state_dict, strict=False)
    del state_dict
sg = sg.to(cfg.device)
sg.train()

# because image generator has no gradient on semantic branch, so leave it here for convenience
g_optim = torch.optim.Adam(sg.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
#sg_optim = torch.optim.Adam(sg.semantic_branch.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
id_optim = torch.optim.Adam(image_disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.9))
sd_optim = torch.optim.Adam(seg_disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.9))

softmax = torch.nn.Softmax2d()
softmax = softmax.cuda()

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

record = {
    "image_disc_loss": [],
    "image_disc_gp": [],
    "seg_disc_loss": [],
    "seg_disc_gp": [],
    "image_gen_loss": [],
    "seg_gen_loss": []}

pbar = tqdm(utils.infinite_dataloader(cfg.dl, cfg.n_iter + 1), total=cfg.n_iter)
for ind, sample in enumerate(pbar):
    ind += 1
    real_image, real_label_compat = sample
    real_image = real_image.cuda()
    real_label = utils.onehot(real_label_compat, cfg.n_class).cuda()
    real_label = F.interpolate(real_label, real_image.size(2), mode="bilinear", align_corners=True)
    latent.normal_()

    with torch.no_grad():
        fake_image = sg(latent, seg=False).clamp(-1, 1)
        fake_label_logit = sg.extract_segmentation(sg.stage)[0]
        fake_label = softmax(fake_label_logit)

    image_disc_loss, image_disc_gp = wgan_gp(image_disc, real_image, fake_image)
    id_optim.step()
    seg_disc_loss, seg_disc_gp = wgan_gp(seg_disc, real_label, fake_label)
    sd_optim.step()

    # Train gen
    sg.zero_grad()
    utils.requires_grad(image_disc, False)
    utils.requires_grad(seg_disc, False)
    latent.normal_()
    # detach for prevent semantic branch's gradient to backbone
    gen, gen_label_logit = sg(latent)
    gen_label = softmax(gen_label_logit)
    image_disc_gen = image_disc(gen)
    image_gen_loss = -image_disc_gen.mean()
    seg_disc_gen = seg_disc(gen_label)
    seg_gen_loss = -seg_disc_gen.mean()
    (image_gen_loss + seg_gen_loss).backward()
    g_optim.step()
    utils.requires_grad(image_disc, True)
    utils.requires_grad(seg_disc, True)

    # display
    record['image_disc_loss'].append(utils.torch2numpy(image_disc_loss))
    record['seg_disc_loss'].append(utils.torch2numpy(seg_disc_loss))
    record['image_disc_gp'].append(utils.torch2numpy(image_disc_gp))
    record['seg_disc_gp'].append(utils.torch2numpy(seg_disc_gp))
    record['image_gen_loss'].append(utils.torch2numpy(image_gen_loss))
    record['seg_gen_loss'].append(utils.torch2numpy(seg_gen_loss))

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

        p = next(image_disc.parameters())
        print("Image Disc: %f %f" % (p.max(), p.min()))
        p = next(seg_disc.parameters())
        print("Seg Disc: %f %f" % (p.max(), p.min()))
        p = next(sg.parameters())
        print("Gen: %f %f" % (p.max(), p.min()))

    if ind % cfg.save_iter == 0 or ind == cfg.n_iter:
        print(f"=> Snapshot model {ind}")
        torch.save(sg.state_dict(), cfg.expr_dir + "/gen_iter_%06d.model" % ind)
        torch.save(image_disc.state_dict(), cfg.expr_dir + "/image_disc_iter_%06d.model" % ind)
        torch.save(seg_disc.state_dict(), cfg.expr_dir + "/seg_disc_iter_%06d.model" % ind)

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        if cfg.seg > 0:
            real_label_viz = []
            num = min(4, real_label.shape[0])
            for i in range(num):
                img = (real_image[i] + 1) / 2
                viz = utils.tensor2label(real_label_compat[i], cfg.n_class)
                real_label_viz.extend([img, viz])
            real_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear", align_corners=True).cpu() for m in real_label_viz])

            fake_label_viz = []
            fake_label_compat = fake_label_logit.argmax(1)
            for i in range(num):
                img = (fake_image[i] + 1) / 2
                viz = utils.tensor2label(fake_label_compat[i], cfg.n_class)
                fake_label_viz.extend([img, viz])
            fake_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear", align_corners=True).cpu() for m in fake_label_viz])

            vutils.save_image(real_label_viz, cfg.expr_dir + "/real_label_viz_%05d.png" % ind, nrow=2)
            vutils.save_image(fake_label_viz, cfg.expr_dir + "/fake_label_viz_%05d.png" % ind, nrow=2)
        else:
            vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
            vutils.save_image(real_image[:4], cfg.expr_dir + '/real_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")