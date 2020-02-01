from home import utils
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
    return (mask * ce).mean()


def mask_mse_loss(mask, x, y):
    mseloss = ((x - y) * mask) ** 2
    return mseloss.sum() / mask.sum()


def baseline_edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=5, lr=1e-2):
    T = Temperture()
    latent = latent.detach().clone()
    latent.requires_grad = True
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
    T = Temperture()
    latent = latent.detach().clone()
    latent.requires_grad = True
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
    T = Temperture()
    latent = latent.detach().clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
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


def celossreg_external_edit_image_stroke_slow(external_model, model, latent, noises, image_stroke, image_mask, n_iter=5, lr=1e-2, n_reg=5):
    latent = latent.detach().clone()
    latent.requires_grad = True
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
    latent = latent.detach().clone()
    latent.requires_grad = True
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

