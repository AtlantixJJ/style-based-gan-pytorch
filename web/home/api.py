"""
API for interaction between GAN and django views
By Jianjin Xu.
8/25/2019
"""
import sys
sys.path.insert(0, '..')
import importlib
import json
import os
import random
import numpy as np
from PIL import Image
from datetime import datetime
from home.colorize import Colorize

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


def stroke2array(image, target_size=None):
    image = image.convert('RGBA')
    if target_size is not None:
        image = image.resize(target_size)
    w, h = image.size
    origin = np.zeros([w, h, 3])
    mask = np.zeros([w, h])
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

    return [origin, mask]


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

# test function


def generate_random_image(size):
    color = '#' + ''.join(random.sample('0123456789ABCDEF', 8))
    background = Image.new('RGBA', size, color)
    return background


class ImageGenerationAPI(object):
    def update_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        self.models_config = self.config['models']
        # for name, mc in self.models_config.items():
        #     mc['mode'] = 'RGB' if mc['out_dim'] == 3 else 'L'
        # self.imsize = self.config['image_size']
        self.data_dir = self.config['collect_data_dir']

    def init_model(self):
        self.models = {}
        self.colorizer = {}
        for name, mc in self.models_config.items():
            ind = mc['model_def'].rfind('.')
            module_name = mc['model_def'][:ind]
            obj_name = mc['model_def'][ind+1:]
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(mc['model_args']['gpu'])
            self.models[name] = obj(**mc['model_args'])
            self.colorizer[name] = Colorize(self.models[name].n_class)

    def __init__(self, config_file):
        self.config_file = config_file
        self.update_config()
        self.init_model()

    def has_model(self, model_name):
        return model_name in list(self.models.keys())

    def debug_mask_image(self, model_name, mask, latent):
        latent = np.fromstring(latent, dtype=np.float32).reshape((1, -1))
        image, label = self.models[model_name](latent)
        latent = np.float32(latent).tobytes()
        image = std_img_shape(image)
        image = np.uint8(image)
        save_image_with_time(self.data_dir, image, "gen")
        save_image_with_time(self.data_dir, mask, "mask")
        return image, latent

    def generate_new_image(self, model_name):
        latent_size = self.models_config[model_name]['in_dim']
        latent = np.random.normal(0, 2, (1, latent_size)).astype('float32')
        image, label = self.models[model_name](latent)
        label_viz = self.colorizer[model_name](label[0].detach().cpu().numpy())
        latent = np.float32(latent).tobytes()
        image = std_img_shape(image)
        image = np.uint8(image)
        save_image_with_time(self.data_dir, image, "gen")
        save_image_with_time(self.data_dir, label_viz, "seg")
        return image, label_viz, latent
