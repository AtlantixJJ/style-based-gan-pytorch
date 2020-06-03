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
import torch
from home.colorize import Colorize
from home.utils import *

CELEBA_COLORS = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),(64, 128, 128),(192, 128, 128)]

class ImageGenerationAPI(object):
    def update_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        self.models_config = self.config['models']
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

    def generate_image_given_stroke(self, model_name, latent, noise,
        image_stroke, image_mask, label_stroke, label_mask):
        save_image_with_time(self.data_dir, image_stroke, "image_stroke")
        save_image_with_time(self.data_dir, label_stroke, "label_stroke")
        save_image_with_time(self.data_dir, image_mask, "image_mask")
        save_image_with_time(self.data_dir, label_mask, "label_mask")
        
        latent      = np.fromstring(latent, dtype=np.float32).reshape((1, -1))
        noise       = np.fromstring(noise, dtype=np.float32).reshape((-1,))
        save_npy_with_time(self.data_dir, latent, "origin_latent")
        save_npy_with_time(self.data_dir, noise, "origin_noise")
        
        size = self.models_config[model_name]["output_size"]
        model = self.models[model_name]
        device = model.device
        latent      = torch.from_numpy(latent).to(device)
        noise       = torch.from_numpy(noise).to(device)
        image_stroke= preprocess_image(image_stroke, size).to(device)
        image_mask  = preprocess_mask(image_mask, size).to(device)
        x = torch.from_numpy(imresize(label_stroke, (size, size)))
        t = torch.zeros(size, size)
        for i, c in enumerate(CELEBA_COLORS):
            t[color_mask(x, c)] = i
        label_stroke = t.unsqueeze(0).to(device)
        label_mask  = preprocess_mask(label_mask, size).squeeze(1).to(device)
        
        res = 0
        if label_mask.sum() < 1:
            print("=> Generate given image stroke")
            res = model.generate_given_image_stroke(
                latent, noise, image_stroke, image_mask)
        else:#elif image_mask.sum() < 1:
            print("=> Generate given label stroke")
            res = model.generate_given_label_stroke(
                latent, noise, label_stroke, label_mask)
        print("=> Generate completed")
        image, label, latent, noise, record = res
        image = image[0]
        label = label[0]
        label_viz = self.colorizer[model_name](label)
        latent = np.float32(latent).tobytes()
        noise = np.float32(noise).tobytes()

        save_image_with_time(self.data_dir, image, "gen") # generated
        save_image_with_time(self.data_dir, label_viz, "seg")
        plot_dic(record)
        save_plot_with_time(self.data_dir, "record")
        return image, label_viz, latent, noise

    def generate_new_image(self, model_name):
        print("=> Generate noise")
        latent, noise = self.models[model_name].generate_noise()
        print("=> Generating")
        image, label = self.models[model_name](latent, noise)
        print("=> Generation completed")
        image = image[0]
        label = label[0]
        label_viz = self.colorizer[model_name](label)
        latent = np.float32(latent.cpu()).tobytes()
        noise = np.float32(noise.cpu()).tobytes()
        return image, label_viz, latent, noise
