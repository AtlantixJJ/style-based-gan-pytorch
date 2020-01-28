import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
from datetime import datetime


def preprocess_image(arr, size=(1024, 1024)):
    """
    arr in [0, 255], shape (H, W, C)
    """
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    t = (t - 127.5) / 127.5
    t = F.interpolate(t, size=size, mode="bilinear")
    return t


def preprocess_mask(mask, size=(1024, 1024)):
    """
    mask in [0, 255], shape (H, W)
    """
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    t = t / 255.
    t = F.interpolate(t, size=size, mode="bilinear")
    return t


def stroke2array(image, target_size=None):
    image = image.convert('RGBA')
    if target_size is not None:
        image = image.resize(target_size)
    w, h = image.size
    origin = np.zeros([w, h, 3], dtype="uint8")
    mask = np.zeros([w, h], dtype="uint8")
    mask_image = Image.new('L', (w, h))
    new_image = Image.alpha_composite(
        Image.new('RGBA', (w, h), 'white'), image)

    for i in range(w):
        for j in range(h):
            masked = image.getpixel((i, j))[3] > 0
            color = new_image.getpixel((i, j))
            origin[j, i] = color[:3]
            mask[j, i] = masked
            mask_image.putpixel(
                (i, j), int(masked * 255))
            new_image.putpixel(
                (i, j), (color[0], color[1], color[2], int(masked * 255)))

    return origin, mask


def std_img_shape(x):
    """
    Standardize image shape
    """
    if len(x.shape) > 3:
        x = x[0]
    
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)

    if x.shape[-1] == 1:
        x = x[:, :, 0]
    return x


def open_image(name):
    with open(name, "rb") as f:
        return np.asarray(Image.open(f))


def save_image(name, image):
    fmt = "JPEG" if ".jpg" in name else "PNG"
    with open(name, "wb") as f:
        Image.fromarray(image).convert("RGB").save(f, format=fmt)


def save_image_with_time(dirname, image, name):
    """
    Args:
        image: numpy image
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    save_image(os.path.join(dirname, '%s_%s.png' %
                            (time_str, name)), image)


def save_plot_with_time(dirname, name):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    fpath = os.path.join(dirname, '%s_%s.png' % (time_str, name))
    plt.savefig(fpath)
    plt.close()


def copy_tensor(dst, src):
    dst.requires_grad = False
    dst.copy_(src)
    dst.requires_grad = True


def torch2numpy(x):
    try:
        return x.detach().cpu().numpy()
    except:
        return x


def plot_dic(dic, file=None):
    n = len(dic.items())
    fig = plt.figure(figsize=(3, 3 * n))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(1, n, i + 1)
        ax.plot(v)
        ax.legend([k])
    if file is not None:
        plt.savefig(file)
        plt.close()
