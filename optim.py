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
    latent = latent.detach().clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    orig_image, orig_seg = model(latent)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)
    target_label = orig_label * (1 - label_mask) + label_stroke * label_mask
    target_label = target_label.long()

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)
        current_label = seg.argmax(1)
        diff_mask = (current_label != target_label).float()
        total_diff = diff_mask.sum()
        mseloss = mask_mse_loss(diff_mask, image, orig_image)
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


def extended_latent_edit_label_stroke(model, latent, noises, label_stroke, label_mask,
    n_iter=5, lr=1e-2):
    latents = latent.detach().split(1, dim=1)
    # 18 x (1, 512)
    for l in latents:
        l.requires_grad = True
    latent = torch.cat(latents, dim=1)
    optim = torch.optim.Adam([latents[0]], lr=lr)
    clr = lr
    for l in latents[1:]:
        clr *= 0.5
        optim.add_param_group({"params": l, "lr": clr})
    model.set_noise(noises)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    orig_image, orig_seg = model(latent)
    orig_image = orig_image.detach().clone()
    orig_label = orig_seg.argmax(1)
    target_label = orig_label * (1 - label_mask) + label_stroke * label_mask
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
        print(grad.shape, grad.min(), grad.max())
        grad_norm = torch.norm(grad.view(-1), 2)
        for i in range(len(latents)):
            latents[i].grad = grad[:, i]
        optim.step()

        record["segdiff"].append(utils.torch2numpy(total_diff))
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image, seg = model(latent)
    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return image, label, latent, noises, record


def baseline_edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=5, lr=1e-2):
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

