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
s = str(cfg)
cfg.setup()
print(s)
with open(cfg.expr_dir + "/config.txt", "w") as f:
	f.write(s)

# global variables
upsample = int(np.log2(cfg.imsize // 4))
latent = torch.randn(cfg.batch_size, 128).cuda()
eps = torch.rand(cfg.batch_size, 1, 1, 1).cuda()

# global utilities
softmax = torch.nn.Softmax2d().cuda()
logsoftmax = torch.nn.CrossEntropyLoss().cuda()

# ordinary discriminator
disc = model.simple.Discriminator(upsample=upsample)
if len(cfg.disc_load_path) > 0:
    state_dict = torch.load(cfg.disc_load_path, map_location='cpu')
    disc.load_state_dict(state_dict)
    del state_dict
disc = disc.to(cfg.device)
disc.train()

# ordinary generator
generator = model.simple.Generator(upsample=upsample)
if len(cfg.gen_load_path) > 0:
    state_dict = torch.load(cfg.gen_load_path, map_location='cpu')
    generator.load_state_dict(state_dict)
    del state_dict
generator = generator.to(cfg.device)
generator.train()

# internal segmentation model
with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]
sep_model = model.semantic_extractor.LinearSemanticExtractor(
    n_class=cfg.n_class,
    dims=dims).cuda().train()

# external model
model_path = "checkpoint/faceparse_unet_128.pth"
state_dict = torch.load(model_path, map_location='cpu')
external_model = unet.unet(train_size=128)
external_model.load_state_dict(state_dict)
external_model.cuda().eval()
utils.requires_grad(external_model, False)

# optimizer
g_optim = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
d_optim = torch.optim.Adam(disc.parameters(), lr=cfg.lr * 2, betas=(0.0, 0.9))

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
for ind, sample in enumerate(pbar):
    ind += 1
    real_image, real_label_compat = sample
    real_image = real_image.cuda()
    real_label = utils.onehot(real_label_compat, cfg.n_class).cuda()
    latent.normal_()

    with torch.no_grad():
        fake_image, stage = generator.get_stage(latent)
        fake_label = sep_model(stage)[0].argmax(1)

    # Train disc
    disc_loss, grad_penalty = wgan_gp(disc, real_image, fake_image)
    d_optim.step()

    # Train gen
    generator.zero_grad()
    utils.requires_grad(disc, False)
    latent.normal_()
    # detach for prevent semantic branch's gradient to backbone
    gen, stage = generator.get_stage(latent, detach=True)
    gen_label_logit = sep_model(stage)[0]
    gen_label = gen_label_logit.argmax(1)
    # discriminative loss
    gen_loss = -disc(gen).mean()
    # image-inconsistent loss
    image_loss = logsoftmax(external_model(gen), gen_label)
    # total loss
    (gen_loss + image_loss).backward()
    g_optim.step()
    # label-inconsistent loss
    sep_model.zero_grad()
    with torch.no_grad():
        gen = generator.get_stage(latent, detach=True, seg=False)
    gen_label_logit = gen.extract_segmentation(generator.stage)[0]
    label = external_model(gen).argmax(1)
    label_loss = logsoftmax(gen_label_logit, label)
    sep_model.optim.step()
    utils.requires_grad(disc, True)

    # display
    record['disc_loss'].append(utils.torch2numpy(disc_loss))
    record['grad_penalty'].append(utils.torch2numpy(grad_penalty))
    record['gen_loss'].append(utils.torch2numpy(gen_loss))
    record['image_loss'].append(utils.torch2numpy(image_loss))
    record['label_loss'].append(utils.torch2numpy(label_loss))

    if cfg.debug:
        print(record.keys())
        l = []
        for k in record.keys():
            l.append(record[k][-1])
        print(l)

        p = next(disc.parameters())
        print("Disc: %f %f" % (p.max(), p.min()))
        p = next(generator.parameters())
        print("Gen: %f %f" % (p.max(), p.min()))

    if ind % cfg.save_iter == 0 or ind == cfg.n_iter:
        print(f"=> Snapshot model {ind}")
        dic = {
            "gen" : generator.state_dict(),
            "disc": disc.state_dict(),
            "sep" : sep_model.state_dict()}
        torch.save(dic, cfg.expr_dir + "/iter_%06d.model" % ind)

    if ind % cfg.disp_iter == 0 or ind == cfg.n_iter or cfg.debug:
        if cfg.seg > 0:
            fake_label_viz = []
            for i in range(num):
                img = (fake_image[i] + 1) / 2
                viz = utils.tensor2label(fake_label[i], cfg.n_class)
                fake_label_viz.extend([img, viz])
            fake_label_viz = torch.cat([F.interpolate(m.unsqueeze(0), 256, mode="bilinear", align_corners=True).cpu() for m in fake_label_viz])

            vutils.save_image(fake_label_viz, cfg.expr_dir + "/fake_label_viz_%05d.png" % ind, nrow=2)
        else:
            vutils.save_image(gen[:4], cfg.expr_dir + '/gen_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
            vutils.save_image(real_image[:4], cfg.expr_dir + '/real_%06d.png' % ind,
                                nrow=2, normalize=True, range=(-1, 1))
        utils.write_log(cfg.expr_dir, record)
        utils.plot_dic(record, "loss", cfg.expr_dir + "/loss.png")