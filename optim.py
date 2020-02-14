import utils
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math


class Temperture(object):
    def __init__(self, T0=0.1, alpha=0.04):
        self.T0 = T0
        self.alpha = alpha
    
    def __call__(self, k):
        return math.exp(-self.alpha * k) * self.T0


def mask_cross_entropy_loss(mask, x, y): # requires more than editing need
    ce = F.cross_entropy(x, y, reduction="none")
    return (mask * ce).sum() / mask.sum()


def mask_mse_loss(mask, x, y):
    mseloss = ((x - y) * mask) ** 2
    return mseloss.sum() / mask.sum()


"""
Return the EL and optimization variable
"""
def get_el_from_latent(latent, mapping_network, method):
    el = 0
    # latent is assumed to be (1, 512) noises or (18, 512) for ML
    if "LL" in method:
        gl = mapping_network(latent)
        el = gl.expand(18, -1).unsqueeze(0)
    elif "GL" in method:
        el = latent.expand(-1, 18, -1)
    elif "ML" in method:
        mgl = [mapping_network(z.unsqueeze(0)) for z in latent]
        el = torch.stack(mgl, dim=1)
    elif "EL" in method:
        el = latent
    return el


def get_image_seg_celeba(model, el, external_model, method):
    if "internal" in method:
        return model(el)
    elif "external" in method:
        image = model(el, seg=False)
        # [NOTICE]: This is hardcode for CelebA
        seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
        return image, seg


def edit_label_stroke(model, latent, noises, label_stroke, label_mask,
    n_iter=5, n_reg=5, lr=1e-2, method="celossreg-label-ML-internal", external_model=None, mapping_network=None):
    latent = latent.detach().clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "gradnorm": []}
    if "celossreg" in method:
        record.update({"celoss": [], "segdiff": []})
        n_reg = 0 # no regularization in baseline method

    el = get_el_from_latent(latent, mapping_network, method)
    orig_image, orig_seg = get_image_seg_celeba(model, el, external_model, method)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)
    target_label = orig_label.float() * (1 - label_mask) + label_stroke * label_mask
    target_label = target_label.long()

    for _ in tqdm(range(n_iter)):
        el = get_el_from_latent(latent, mapping_network, method)
        image, seg = get_image_seg_celeba(model, el, external_model, method)

        if "celossreg" in method:
            current_label = seg.argmax(1)
            diff_mask = (current_label != target_label).float()
            total_diff = diff_mask.sum()
            celoss = mask_cross_entropy_loss(diff_mask, seg, target_label)
            record["segdiff"].append(utils.torch2numpy(total_diff))
            record["celoss"].append(utils.torch2numpy(celoss))
        else:
            celoss = 0

        mseloss = mask_mse_loss(1 - diff_mask, image, orig_image)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        grad_norm = torch.norm(grad.view(-1), 2)
        latent.grad = grad
        optim.step()

        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

        # celoss regularization
        for _ in range(n_reg):
            el = get_el_from_latent(latent, mapping_network, method)
            image, seg = get_image_seg_celeba(model, el, external_model, method)
            revise_label = seg.argmax(1).long()
            # directly use cross entropy may also decrease other part
            diff_mask = (revise_label != orig_label).float()
            total_diff = diff_mask.sum()
            celoss = mask_cross_entropy_loss(diff_mask, seg, orig_label)
            grad = torch.autograd.grad(celoss, latent)[0]
            grad_norm = torch.norm(grad[0], 2)
            latent.grad = grad
            optim.step()
            record["celoss"].append(utils.torch2numpy(celoss))
            record["mseloss"].append(utils.torch2numpy(mseloss))
            record["segdiff"].append(total_diff)
            record["gradnorm"].append(utils.torch2numpy(grad_norm))

    with torch.no_grad():
        el = get_el_from_latent(latent, mapping_network, method)
        image, seg = get_image_seg_celeba(model, el, external_model, method)
        image = (1 + image.clamp(-1, 1)) / 2
        label = seg.argmax(1)
    return image, label, latent, noises, record


def edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=5, n_reg=5, lr=1e-2, method="celossreg-label-ML-internal", external_model=None, mapping_network=None):
    latent = latent.detach().clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "gradnorm": []}
    if "celossreg" in method:
        record.update({"celoss": [], "segdiff": []})
        n_reg = 0 # no regularization in baseline method
    
    el = get_el_from_latent(latent, mapping_network, method)
    orig_image, orig_seg = get_image_seg_celeba(model, el, external_model, method)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)

    for _ in tqdm(range(n_iter)):
        el = get_el_from_latent(latent, mapping_network, method)
        image, seg = get_image_seg_celeba(model, el, external_model, method)

        if "celossreg" in method:
            current_label = seg.argmax(1)
            diff_mask = (current_label != orig_label).float()
            total_diff = diff_mask.sum()
            celoss = mask_cross_entropy_loss(diff_mask, seg, orig_label)
            record["segdiff"].append(utils.torch2numpy(total_diff))
            record["celoss"].append(utils.torch2numpy(celoss))
        else:
            celoss = 0

        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()

        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    with torch.no_grad():
        el = get_el_from_latent(latent, mapping_network, method)
        image, seg = get_image_seg_celeba(model, el, external_model, method)
        image = (1 + image.clamp(-1, 1)) / 2
        label = seg.argmax(1)
    return image, label, latent, noises, record