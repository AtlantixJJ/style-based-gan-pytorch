import utils
import torch
import torch.nn.functional as F
from tqdm import tqdm


def baseline_edit_image_stroke(model, latent, noises, image_stroke, image_mask, n_iter=20, lr=1e-2):
    latent = latent.clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, orig_seg = model(latent)
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "gradnorm": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)

        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        grad = torch.autograd.grad(mseloss, latent)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()

        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return orig_image, orig_label, image, label, latent, noises, record


def celossreg_edit_image_stroke_slow(model, latent, noises, image_stroke, image_mask,
    n_iter=20, lr=1e-2, n_reg=5):
    latent = latent.clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, orig_seg = model(latent)
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)

        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        latent.grad = torch.autograd.grad(mseloss, latent)[0]
        optim.step()

        count = 0
        total_diff = 0
        while count < n_reg:
            image, seg = model(latent)
            mseloss = ((image_stroke - image) * image_mask) ** 2
            mseloss = mseloss.sum() / image_mask.sum()
            revise_label = seg.argmax(1).long()
            celoss = 10 * logsoftmax(seg, orig_label)
            # directly use cross entropy may also decrease other part
            diff_mask = (revise_label != orig_label).float()
            total_diff = diff_mask.sum()
            count += 1
            grad_seg = torch.autograd.grad(celoss, seg)[0]
            grad = torch.autograd.grad(seg, latent,
                grad_outputs=grad_seg * diff_mask)[0]
            grad_norm = torch.norm(grad[0], 2)
            latent.grad = grad
            optim.step()
            record["celoss"].append(utils.torch2numpy(celoss))
            record["mseloss"].append(utils.torch2numpy(mseloss))
            record["segdiff"].append(total_diff)
            record["gradnorm"].append(utils.torch2numpy(grad_norm))
            if total_diff < 10:
                break

    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return orig_image, orig_label, image, label, latent, noises, record


def celossreg_edit_image_stroke(model, latent, noises, image_stroke, image_mask,
    n_iter=20, lr=1e-2):
    latent = latent.clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, orig_seg = model(latent)
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)
        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        revise_label = seg.argmax(1).long()
        celoss = 10 * logsoftmax(seg, orig_label)
        loss = mseloss + celoss
        # directly use cross entropy may also decrease other part
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        grad_seg = torch.autograd.grad(loss, seg)[0]
        grad = torch.autograd.grad(seg, latent,
            grad_outputs=grad_seg * diff_mask)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return orig_image, orig_label, image, label, latent, noises, record


def celossreg_external_edit_image_stroke_slow(external_model, model, latent, noises, image_stroke, image_mask, n_iter=20, lr=1e-2, n_reg=5):
    latent = latent.clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, _ = model(latent)
        orig_seg = utils.diff_idmap(external_model(orig_image))
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, _ = model(latent)

        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        latent.grad = torch.autograd.grad(mseloss, latent)[0]
        optim.step()

        count = 0
        total_diff = 0
        while count < n_reg:
            image, _ = model(latent)
            seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
            mseloss = ((image_stroke - image) * image_mask) ** 2
            mseloss = mseloss.sum() / image_mask.sum()
            revise_label = seg.argmax(1).long()
            celoss = logsoftmax(seg, orig_label)
            # directly use cross entropy may also decrease other part
            diff_mask = (revise_label != orig_label).float()
            total_diff = diff_mask.sum()
            count += 1
            grad_seg = torch.autograd.grad(celoss, seg)[0]
            grad = torch.autograd.grad(seg, latent,
                grad_outputs=grad_seg * diff_mask)[0]
            grad_norm = torch.norm(grad[0], 2)
            latent.grad = grad
            optim.step()
            record["celoss"].append(utils.torch2numpy(celoss))
            record["mseloss"].append(utils.torch2numpy(mseloss))
            record["segdiff"].append(total_diff)
            record["gradnorm"].append(utils.torch2numpy(grad_norm))
            if total_diff < 10:
                break

    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return orig_image, orig_label, image, label, latent, noises, record


def celossreg_external_edit_image_stroke(external_model, model, latent, noises, image_stroke, image_mask, n_iter=20, lr=1e-2):
    latent = latent.clone()
    latent.requires_grad = True
    optim = torch.optim.Adam([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, _ = model(latent)
        orig_seg = utils.diff_idmap(external_model(orig_image))
    orig_image = (1 + orig_image.clamp(-1, 1)) / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": [], "gradnorm": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, _ = model(latent)
        seg = utils.diff_idmap(external_model(image.clamp(-1, 1)))
        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        revise_label = seg.argmax(1).long()
        celoss = logsoftmax(seg, orig_label)
        loss = mseloss + celoss
        # directly use cross entropy may also decrease other part
        diff_mask = (revise_label != orig_label).float()
        total_diff = diff_mask.sum()
        grad_seg = torch.autograd.grad(loss, seg)[0]
        grad = torch.autograd.grad(seg, latent,
            grad_outputs=grad_seg * diff_mask)[0]
        grad_norm = torch.norm(grad[0], 2)
        latent.grad = grad
        optim.step()
        record["celoss"].append(utils.torch2numpy(celoss))
        record["mseloss"].append(utils.torch2numpy(mseloss))
        record["segdiff"].append(total_diff)
        record["gradnorm"].append(utils.torch2numpy(grad_norm))

    image = (1 + image.clamp(-1, 1)) / 2
    label = seg.argmax(1)
    return orig_image, orig_label, image, label, latent, noises, record


def get_optim(t, **kwargs):
    func = 0
    dic = {}
    if t == "baseline-latent":
        dic = {k:v for k,v in kwargs.items() if k != "external_model"}
        func = baseline_edit_image_stroke
    elif t in ["celossreg-latent", "celossreg-extended-latent"]:
        dic = {k:v for k,v in kwargs.items() if k != "external_model"}
        func = celossreg_edit_image_stroke
    elif t == ["celossregexternal-latent", "celossregexternal-extened-latent"]:
        dic = kwargs
        func = celossreg_external_edit_image_stroke
    elif t == "celossreg-latent-slow":
        dic = {k:v for k,v in kwargs.items() if k != "external_model"}
        func = celossreg_edit_image_stroke_slow
    elif t == "celossregexternal-latent-slow":
        dic = kwargs
        func = celossreg_external_edit_image_stroke_slow
    return func(**dic)
