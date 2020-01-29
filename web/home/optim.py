from home import utils
import torch
from tqdm import tqdm


def baseline_edit_image_stroke(model, latent, noises, image_stroke, image_mask, n_iter=20, lr=1e-2):
    latent.requires_grad = True
    optim = torch.optim.SGD([latent], lr=lr)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, orig_seg = model(latent)
    orig_image = (1 + orig_image.clamp(-1, 1)) * 255 / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": []}

    if image_mask.sum() < 1.0:
        return orig_image, orig_label, latent, noises, record

    for _ in tqdm(range(n_iter)):
        image, seg = model(latent)

        mseloss = ((image_stroke - image) * image_mask) ** 2
        mseloss = mseloss.sum() / image_mask.sum()
        latent.grad = torch.autograd.grad(mseloss, latent)[0]
        optim.step()

        record["mseloss"].append(utils.torch2numpy(mseloss))

    image = (1 + image.clamp(-1, 1)) * 255 / 2
    label = seg.argmax(1)
    return orig_image, image, label, latent, noises, record


def celossreg_edit_image_stroke(model, latent, noises, image_stroke, image_mask, n_iter=20, lr=1e-2):
    latent.requires_grad = True
    optim = torch.optim.SGD([latent], lr=lr)
    logsoftmax = torch.nn.CrossEntropyLoss().to(latent.device)
    model.set_noise(noises)
    with torch.no_grad():
        orig_image, orig_seg = model(latent)
    orig_image = (1 + orig_image.clamp(-1, 1)) * 255 / 2
    orig_label = orig_seg.argmax(1)
    record = {"mseloss": [], "celoss": [], "segdiff": []}

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
        while count < 10:
            image, seg = model(latent)
            mseloss = ((image_stroke - image) * image_mask) ** 2
            mseloss = mseloss.sum() / image_mask.sum()
            revise_label = seg.argmax(1).long()
            celoss = logsoftmax(seg, orig_label)
            # directly use cross entropy may also decrease other part
            diff_mask = (revise_label != orig_label).float()
            total_diff = diff_mask.sum()
            count += 1
            record["celoss"].append(utils.torch2numpy(celoss))
            record["mseloss"].append(utils.torch2numpy(mseloss))
            record["segdiff"].append(total_diff)
            if total_diff < 100:
                break
            grad_seg = torch.autograd.grad(celoss, seg)[0]
            latent.grad = torch.autograd.grad(seg, latent,
                grad_outputs=grad_seg * diff_mask)[0]
            optim.step()

    image = (1 + image.clamp(-1, 1)) * 255 / 2
    label = seg.argmax(1)
    return orig_image, image, label, latent, noises, record


def get_optim(t):
    if t == "baseline-latent":
        return baseline_edit_image_stroke
    elif t == "celossreg-latent":
        return celossreg_edit_image_stroke
