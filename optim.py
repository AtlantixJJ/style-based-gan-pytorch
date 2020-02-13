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


def baseline_edit_label_stroke(model, latent, noises, label_stroke, label_mask,
    n_iter=5, lr=1e-2):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    orig_image, orig_seg = model(latent)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)
    target_label = orig_label.float() * (1 - label_mask) + label_stroke * label_mask
    target_label = target_label.long()

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)
        current_label = seg.argmax(1)
        diff_mask = (current_label != target_label).float()
        total_diff = diff_mask.sum()
        mseloss = mask_mse_loss(1 - diff_mask, image, orig_image)
        celoss = mask_cross_entropy_loss(diff_mask, seg, target_label)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad# + T(i) * grad.std() * torch.randn_like(grad)
        optim.step()

        record["segdiff"].append(utils.torch2numpy(total_diff))
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


"""
Return the EL and optimization variable
"""
def get_el_from_latent(latent, mapping_network, method):
    el = 0
    v = 0
    # latent is assumed to be (1, 512) noises or (18, 512) for ML
    if "LL" in method:
        gl = mapping_network(latent)
        el = gl.expand(-1, 18, -1)
        v = latent
    elif "GL" in method:
        el = latent.expand(-1, 18, -1)
        v = latent
    elif "ML" in method:
        mgl = [mapping_network(z) for z in latent]
        el = torch.stack(mgl, dim=1)
        v = latent
    elif "EL" in method:
        el = latent
        v = latent
    return el, v


def get_image_seg_celeba(model, el, external_model, method):
    if "internal" in method:
        return model(el)
    elif "external" in method:
        image = model(el, seg=False)
        # [NOTICE]: This is hardcode for CelebA
        seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
        return image, seg


def edit_label_stroke(model, latent, noises, label_stroke, label_mask,
    n_iter=5, n_reg=5, lr=1e-2, method="ML-internal", external_model=None, mapping_network=None):
    el, var = get_el_from_latent(latent, mapping_network, method)
    optim = torch.optim.Adam([var], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    orig_image, orig_seg = model(latent)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)
    target_label = orig_label.float() * (1 - label_mask) + label_stroke * label_mask
    target_label = target_label.long()

    for ind in tqdm(range(n_iter)):
        image, seg = get_image_seg_celeba(model, el, external_model, method)

        current_label = seg.argmax(1)
        diff_mask = (current_label != target_label).float()
        total_diff = diff_mask.sum()
        mseloss = mask_mse_loss(1 - diff_mask, image, orig_image)
        celoss = mask_cross_entropy_loss(diff_mask, seg, target_label)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        grad_norm = torch.norm(grad.view(-1), 2)
        latent.grad = grad
        optim.step()

        record["segdiff"].append(utils.torch2numpy(total_diff))
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

        for _ in range(n_reg):
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

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def baseline_edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=5, lr=1e-2):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "gradnorm": []}

    for i in tqdm(range(n_iter)):
        image, seg = model(latent)

        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        grad = torch.autograd.grad(mseloss, latent)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad# + T(i) * grad.std() * torch.randn_like(grad)
        optim.step()

        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def celossreg_edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=5, lr=1e-2):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    orig_label = model(latent)[1].argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    for i in tqdm(range(n_iter)):
        image, seg = model(latent)
        revise_label = seg.argmax(1).long()
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        # directly use cross entropy may also decrease other part
        celoss = mask_cross_entropy_loss(diff_mask, seg, orig_label)
        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        #grad += T(i) * grad.std() * torch.randn_like(grad)
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def celossreg_external_edit_image_stroke(external_model, model, latent, noises, image_stroke, image_mask, n_iter=5, lr=1e-2):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    orig_label = model(latent)[1].argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    for i in tqdm(range(n_iter)):
        image, _ = model(latent)
        seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
        revise_label = seg.argmax(1).long()
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        # directly use cross entropy may also decrease other part
        celoss = mask_cross_entropy_loss(diff_mask, seg, orig_label)
        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        loss = mseloss + celoss
        grad = torch.autograd.grad(loss, latent)[0]
        #grad += T(i) * grad.std() * torch.randn_like(grad)
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image, _ = model(latent)
    seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def get_optim_func(t):
    if "baseline" in t:
        func = baseline_edit_image_stroke
    elif "external" in t and "slow" in t:
        func = celossreg_external_edit_image_stroke_slow
    elif "slow" in t:
        func = celossreg_edit_image_stroke_slow
    elif "external" in t:
        func = celossreg_external_edit_image_stroke
    elif "celoss" in t:
        func = celossreg_edit_image_stroke
    return func


def get_optim(t, **kwargs):
    func = 0
    dic = {}
    basic_keys = ["model", "latent", "noises", "image_stroke", "image_mask", "n_iter", "lr"]

    if "baseline" in t:
        dic = {k:kwargs[k] for k in basic_keys}
        func = baseline_edit_image_stroke
    elif "external" in t and "slow" in t:
        dic = {k:kwargs[k] for k in basic_keys + ["external_model", "n_reg"]}
        func = celossreg_external_edit_image_stroke_slow
    elif "slow" in t:
        dic = {k:kwargs[k] for k in basic_keys + ["n_reg"]}
        func = celossreg_edit_image_stroke_slow
    elif "external" in t:
        dic = {k:kwargs[k] for k in basic_keys + ["external_model"]}
        func = celossreg_external_edit_image_stroke
    elif "celoss" in t:
        dic = {k:kwargs[k] for k in basic_keys}
        func = celossreg_edit_image_stroke
    
    return func(**dic)


def get_label_optim(t):
    if "extended" in t:
        return extended_latent_edit_label_stroke
    else:
        return baseline_edit_label_stroke


def celossreg_external_edit_image_stroke_slow(external_model, model, latent, noises, image_stroke, image_mask, n_iter=5, lr=1e-2, n_reg=5):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    orig_label = model(latent)[1].argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    for _ in tqdm(range(n_iter)):
        image, _ = model(latent)

        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        latent.grad = torch.autograd.grad(mseloss, latent)[0]
        optim.step()

        for _ in range(n_reg):
            image, _ = model(latent)
            seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
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

    image, _ = model(latent)
    seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def celossreg_edit_image_stroke_slow(model, latent, noises, image_stroke, image_mask,
    n_iter=5, lr=1e-2, n_reg=5):
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    orig_label = model(latent)[1].argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)

        mseloss = mask_mse_loss(image_mask, image, image_stroke)
        latent.grad = torch.autograd.grad(mseloss, latent)[0]
        optim.step()

        for _ in range(n_reg):
            image, seg = model(latent)
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

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record