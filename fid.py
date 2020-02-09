#!/usr/bin/env python3
import os, sys, glob
sys.path.insert(0, ".")
import pathlib
import numpy as np
import torch, torchvision
from torchvision import transforms
from torchvision import utils as vutils
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from copy import deepcopy
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

import dataset, utils
from model.inception import inception_v3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_feature(model, iterator):
    model.eval()
    features = []
    for sample in tqdm(iterator):
        sample = sample.cuda()
        pred = model(sample)[-1]
        features.append(pred.detach().cpu().numpy().reshape(sample.shape[0], -1))
    return np.concatenate(features)


def calculate_statistics_given_iterator(model, iterator, save_path=None):
    feature = get_feature(model, iterator)
    mu = np.mean(feature, axis=0)
    sigma = np.cov(feature, rowvar=False)
    if save_path is not None:
        np.save(save_path + "_mu_sigma.npy", {"mu": mu, "sigma": sigma})
    return mu, sigma


class GeneratorIterator(object):
    def __init__(self, model, dim=512, tot_num=50000, batch_size=64, cuda=True):
        self.model = model
        self.tot_num = tot_num
        self.dim = dim
        self.cuda = cuda
        self.batch_size = batch_size
        self.num_iter = self.tot_num // self.batch_size
    
    def iterator(self, save_path=None):
        if save_path is not None and not os.path.exists(save_path):
            os.system("mkdir %s" % save_path)

        z = torch.Tensor(self.batch_size, self.dim).cuda()
        if self.num_iter * self.batch_size < self.tot_num:
            self.num_iter += 1
        for i in range(self.num_iter):
            if i == self.num_iter - 1:
                bs = self.tot_num - self.batch_size * i
                if bs < self.batch_size:
                    z = torch.Tensor(bs, self.dim).cuda()
            z = z.normal_()
            t = self.model(z)
            if type(t) is tuple:
                t = t[0]
            t = t.clamp(-1, 1)

            if save_path is not None:
                for idx in range(t.shape[0]):
                    gidx = idx+i*self.batch_size
                    vutils.save_image(
                        (t[idx:idx+1] + 1)/ 2,
                        f"{save_path}/{gidx}.jpg")

            yield utils.tensor_resize_by_pil(t)


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable, pre):
        super(IterableDataset, self).__init__()
        self.iterable = iterable
        self.pre = pre
    
    def __iter__(self):
        #start = 0
        for sample in self.iterable:
            #start += 1
            #vutils.save_image((1 + sample) / 2, f"test/{self.pre}_{start}.png")
            yield sample


class IteratorDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterator):
        super(IteratorDataset, self).__init__()
        self.iterator = iterator
    
    def __iter__(self):
        return self.iterator

        
class PartFIDEvaluator(object):
    def __init__(self, n_class=16):
        self.n_class = n_class
        self.model = inception_v3(pretrained=True, aux_logits=False)
        self.model.eval()
        self.model.cuda()
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def calc_share_feature(self, idx, save_path):
        for i in range(self.n_class):
            if len(self.part_features[i]) == 0:
                continue
            ds = IterableDataset(self.part_features[i], str(i))
            dl = torch.utils.data.DataLoader(ds,
                batch_size=50, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
            feature = get_feature(self.model, dl)
            np.save(f"{save_path}/share_s{idx:02d}_c{i:02d}.npy", feature.astype("float32"))
        del self.part_features
        self.part_features = [[] for _ in range(self.n_class)]

    def large_calc_statistic(self, iterator, save_path):
        self.part_features = [[] for _ in range(self.n_class)]
        # every 500 iteration cost 15G memory
        self.n_share = 0
        for index, data_list in enumerate(tqdm(iterator)):
            for label, image in data_list:
                image_copy = deepcopy(image)
                if image_copy.shape[0] == 1:
                    image_copy = image_copy[0]
                self.part_features[label].append(image_copy)
                del image
            
            if (index + 1) % 1000 == 0:
                print("=> Calculate share %d of feature." % self.n_share)
                self.calc_share_feature(self.n_share, save_path)
                self.n_share += 1
        
        class_number = [len(p) for p in self.part_features]
        if sum(class_number) > 0:
            self.calc_share_feature(self.n_share, save_path)
            self.n_share += 1

    def small_calculate_statistics_given_iterator(self, iterator, save_path=None):
        part_features = [[] for _ in range(self.n_class)]
        # every 500 iteration cost 15G memory
        for index, data_list in enumerate(tqdm(iterator)):
            for label, image in data_list:
                if image.shape[0] == 1:
                    image = image[0]
                part_features[label].append(image)

        # instance number
        self.class_length = [len(p) for p in part_features]
        self.class_mu = []
        self.class_sigma = []

        for i in range(self.n_class):
            if len(part_features[i]) == 0:
                continue
            ds = IterableDataset(part_features[i], str(i))
            dl = torch.utils.data.DataLoader(ds,
                batch_size=50, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
            feature = get_feature(self.model, dl)
            mu = np.mean(feature, axis=0)
            sigma = np.cov(feature, rowvar=False)
            if save_path:
                np.save(f"{save_path}/c{i}_mu_sigma.npy", {"mu": mu, "sigma": sigma})

            
class FIDEvaluator(object):
    def __init__(self, ref_datapath, save_path=None, batch_size=50, cuda=True):
        self.ref_datapath = ref_datapath
        self.save_path = save_path
        self.batch_size = batch_size
        self.cuda = cuda
        self.model = inception_v3(pretrained=True, aux_logits=False)
        self.model.eval()
        if cuda: self.model.cuda()
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def calculate_statistics_given_path(self, path, save_npy=True):
        npy_path = path + "_mu_sigma.npy"
        if os.path.exists(npy_path):
            path = npy_path
            print("=> Use npy file %s" % path)
            f = np.load(path, allow_pickle=True).tolist()
            m, s = f['mu'][:], f['sigma'][:]
            return m, s
        else:
            print("=> Calc from path %s" % path)
            ds = dataset.SimpleDataset(path, (299, 299), self.transform_test)
            #np.random.RandomState(65537).shuffle(ds.files)
            dl = torch.utils.data.DataLoader(ds, batch_size=50, shuffle=False, num_workers=1, pin_memory=True)
            feature = get_feature(self.model, dl)
            mu = np.mean(feature, axis=0)
            sigma = np.cov(feature, rowvar=False)
            if save_npy:
                np.save(path + "_mu_sigma.npy", {"mu": mu, "sigma": sigma})
            return mu, sigma

    def __call__(self, gs, save_path):
        m1, s1 = self.calculate_statistics_given_path(self.ref_datapath)
        m2, s2 = calculate_statistics_given_iterator(
            self.model,
            gs.iterator() if self.save_path is None else gs.iterator(self.save_path),
            save_path)
        return calculate_frechet_distance(m1, s1, m2, s2)


if __name__ == '__main__':
    import model

    name = sys.argv[1]

    stylegan_path = f"checkpoint/karras2019stylegan-{name}-1024x1024.for_g_all.pt"
    celeba_path = "../datasets/CelebAMask-HQ/CelebA-HQ-img"
    save_path = f"../datasets/CelebAMask-HQ/whole_gen_{name}"

    #generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
    generator = model.tf.StyledGenerator()
    generator.load_state_dict(torch.load(stylegan_path), strict=False)
    generator.cuda()

    evaluator = FIDEvaluator(celeba_path, save_path)
    fid_value = evaluator(GeneratorIterator(generator,
        batch_size=1, tot_num=30000, dim=512), save_path)
    print('FID: ', fid_value)