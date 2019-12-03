"""
Semantic enhanced discriminator training.
"""
import sys
sys.path.insert(0, ".")
import os
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
disc = model.disc.Discriminator(from_rgb_activate=True)
disc.load_state_dict(state_dict)
disc.set_semantic_config(cfg.disc_semantic_config)
disc = nn.DataParallel(disc)
disc = disc.to(cfg.device)
disc.train()

state_dict = torch.load(cfg.load_path, map_location='cpu')
sg = model.tfseg.StyledGenerator(semantic=cfg.semantic_config)
sg.load_state_dict(state_dict)
sg = nn.DataParallel(sg)
sg = sg.to(cfg.device)
sg.train()
del state_dict

g_optim = torch.optim.Adam(sg.g_synthesis.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
d_optim = torch.optim.Adam(disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.99))

latent = torch.randn(cfg.batch_size, 512).cuda()
eps = torch.rand(cfg.batch_size, 1, 1, 1).cuda()

pbar = tqdm(utils.infinite_dataloader(cfg.dl, cfg.n_iter + 1), total=cfg.n_iter)
for ind, sample in enumerate(pbar):
    ind += 1
    real_image, real_label_compat = sample
    real_image = real_image.cuda()
    size = real_label_compat.size(1)
    real_label = torch.zeros(cfg.batch_size, cfg.n_class, size, size)
    real_label.scatter_(1, real_label_compat, 1)
    latent.normal_()
    eps.uniform_()

    disc_real_in = torch.cat([real_image, real_label], 1)
    with torch.zero_grad():
        fake_image = sg(latent)
        fake_label_logit = sg.extract_segmentation()[-1]
        fake_label_compat = fake_label_logit.argmax(1, keepdim)
        fake_label = torch.zeros(cfg.batch_size, cfg.n_class, size, size)
        fake_label.scatter_(1, real_label_compat, 1)
    disc_fake_in = torch.cat([fake_image, fake_label], 1)

    # Train disc
    disc.zero_grad()
    disc_real = disc(disc_real_in, step=step, alpha=alpha)
    disc_fake = disc(fake_image, step=step, alpha=alpha)
    disc_loss = disc_fake.mean() - disc_real.mean()
    
    x_hat = eps * real_image.data + (1 - eps) * fake_image.data
    x_hat.requires_grad = True
    disc_x_hat = disc(x_hat, step=step, alpha=alpha)
    grad_x_hat = torch.autograd.grad(
        outputs=disc_x_hat.sum(), 
        inputs=x_hat,
        create_graph=True)[0]
    grad_x_hat_norm = grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1)
    grad_penalty = 10 * ((grad_x_hat_norm - 1) ** 2).mean()
    (disc_loss + grad_penalty).backward()
    d_optim.step()

    # Train gen
    sg.zero_grad()
    utils.requires_grad(disc, False)
    latent.normal_()
    gen = sg(latent)
    gen_label = sg.extract_segmentation()
    disc_gen_in = torch.cat([gen, gen_label], 1)
    disc_gen = disc(disc_gen_in, step=step, alpha=alpha)
    gen_loss = -disc_gen.mean()
    gen_loss.backward()

    gen_loss_val = gen_loss.item()

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

    if ind % cfg.save_iter == 0 or i == cfg.n_iter:
        print("=> Snapshot model %d" % ind)
        torch.save(sg.state_dict(), cfg.expr_dir + "/gen_iter_%06d.model" % ind)
        torch.save(disc.state_dict(), cfg.expr_dir + "/disc_iter_%06d.model" % ind)

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                            nrow=2, normalize=True, range=(-1, 1))
        tarlabels = []
        for i in range(real_label.shape[0]):
            label_viz = utils.tensor2label(real_label[i:i+1], cfg.n_class)
            tarlabels.append(.unsqueeze(0))
        tarviz = torch.cat([F.interpolate(m.float(), 256).cpu() for m in tarlabels])
        genlabels = [utils.tensor2label(s[0], s.shape[1]).unsqueeze(0)
                    for s in segs]
        gen_img = (gen[0:1].clamp(-1, 1) + 1) / 2
        genviz = genlabels + [gen_img]
        genviz = torch.cat([F.interpolate(m.float(), 256).cpu() for m in genviz])
        vutils.save_image(genviz, cfg.expr_dir + "/genlabel_viz_%05d.png" % i, nrow=2)
        vutils.save_image(tarviz, cfg.expr_dir + "/tarlabel_viz_%05d.png" % i, nrow=2)
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, cfg.expr_dir + "/loss.png")

record = cfg.record
noise = [0] * (STEP + 1)
avgmseloss = 0
count = 0
for i in tqdm(range(cfg.n_iter + 1)):
    latent = torch.randn(cfg.batch_size, 512)
    for k in range(STEP + 1):
        size = 4 * 2 ** k
        noise[k] = torch.randn(cfg.batch_size, 1, size, size)

    with torch.no_grad():
        target_image = tg(latent.to(cfg.device1), noise=[n.to(
            cfg.device1) for n in noise], step=STEP, alpha=ALPHA)
        target_image = target_image.detach().to(cfg.device2)

    if i <= cfg.stage2_step:
        lerp_val = lrschedule(i)
        g_optim = g_optim1
    else:
        lerp_val = 1
        g_optim = g_optim2
    set_lerp_val(sg.generator.progression, lerp_val)

    latent2 = latent.to(cfg.device2)
    noise2 = [n.to(cfg.device2) for n in noise]

    gen = sg(latent2, noise=noise2, step=STEP, alpha=ALPHA)
    masks = get_masks(sg.generator.progression)

    if cfg.debug:
        vutils.save_image(visualize_masks(masks), cfg.expr_dir + '/debug1_original_mask.png',
                          nrow=4, normalize=True, range=(0, 1))

    permute_masks(masks)

    if cfg.debug:
        vutils.save_image(visualize_masks(masks), cfg.expr_dir + '/debug2_permuted_mask.png',
                          nrow=4, normalize=True, range=(0, 1))

    gen_sw = sg(latent2, noise=noise2, masks=masks,
                step=STEP, alpha=ALPHA)

    masks = get_masks(sg.generator.progression)

    if cfg.debug:
        vutils.save_image(visualize_masks(masks), cfg.expr_dir + '/debug3_new_mask.png',
                          nrow=4, normalize=True, range=(0, 1))

    mseloss = crit(gen, target_image)
    disc_loss = disc(gen, step=STEP, alpha=ALPHA)

    if cfg.ma > 0:
        mask_avgarea = maskarealoss(
            sg.generator.progression, target=1.0/cfg.att, coef=cfg.ma)
    else:
        mask_avgarea = 0

    if cfg.md > 0:
        mask_divergence = maskdivloss(sg.generator.progression, coef=cfg.md)
    else:
        mask_divergence = 0

    if cfg.mv > 0:
        mask_value = maskvalueloss(
            sg.generator.progression, coef=cfg.mv)
    else:
        mask_value = 0

    loss = mseloss + mask_avgarea + mask_divergence + mask_value + disc_loss.mean()
    with torch.autograd.detect_anomaly():
        loss.backward()
    g_optim.step()
    g_optim.zero_grad()

    mse_data = torch2numpy(mseloss)
    avgmseloss = avgmseloss * 0.9 + mse_data * 0.1
    record['loss'].append(torch2numpy(loss))
    record['lerp_val'].append(lerp_val)
    record['mseloss'].append(mse_data)
    if mask_avgarea > 0:
        record['mask_area'].append(torch2numpy(mask_avgarea))
    if mask_divergence > 0:
        record['mask_div'].append(torch2numpy(mask_divergence))
    if mask_value > 0:
        record['mask_value'].append(torch2numpy(mask_value))

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

    if i % 1000 == 0 and i > 0:
        print("=> Snapshot model %d" % i)
        torch.save(sg.state_dict(), cfg.expr_dir + "/iter_%06d.model" % i)
        vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        vutils.save_image(target_image[:4], cfg.expr_dir + '/target_%06d.png' % i,
                          nrow=2, normalize=True, range=(-1, 1))
        write_log(cfg.expr_dir, record)
