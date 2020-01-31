from __future__ import print_function
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from os.path import join as osj
try:
    import pickle
except:
    import cPickle as pickle
import datetime
import time
import random
from contextlib import contextmanager
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
import glob


class MultiGPUTensor(object):
    def __init__(self, root, n_gpu):
        self.root = root
        self.n_gpu = n_gpu
        self.create_copy()
    
    def create_copy(self):
        self.val = []
        for i in range(self.n_gpu):
            self.val.append(self.root.detach().to("cuda:" + str(i)))
    
    def sync(self):
        for i in range(self.n_gpu):
            self.val[i].copy_(self.root)


######
## Colorization
######


HEATMAP_COLOR = cm.get_cmap("Reds")
def heatmap_numpy(image):
    """
    assume numpy array as input: (N, H, W) in [0, 1]
    returns: (N, H, W, 3)
    """
    return HEATMAP_COLOR(image)[:, :, :, :3]

def heatmap_torch(tensor):
    """
    assume 4D torch.Tensor (N, 1, H, W)
    """
    numpy_arr = torch2numpy(tensor[:, 0, :, :])
    heatmap = HEATMAP_COLOR(numpy_arr)[:, :, :, :3]
    t = torch.from_numpy(heatmap.transpose(0, 3, 1, 2)).float()
    return t.to(tensor.device)


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        if gray_image.shape[0] == 1:
            gray_image = gray_image[0]
        size = gray_image.shape
        assert len(size) == 2
        h, w = size

        if isinstance(gray_image, torch.Tensor):
            color_image = torch.zeros(3, h, w, device=gray_image.device).fill_(0)
            for label in range(0, len(self.cmap)):
                mask = (label == gray_image).cpu()
                color_image[0][mask] = int(self.cmap[label, 0])
                color_image[1][mask] = int(self.cmap[label, 1])
                color_image[2][mask] = int(self.cmap[label, 2])
        else:
            color_image = np.zeros((h, w, 3), dtype="uint8")
            for label in range(len(self.cmap)):
                mask = (label == gray_image)
                color_image[mask] = self.cmap[label]
        return color_image


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    """
    edge_num = int(np.ceil(np.power(N + , 1/3))) - 1
    cmap = np.zeros((N, 3), dtype=np.uint8)
    step_size = 255. / edge_num
    cmap[0] = (0, 0, 0)
    count = 1
    for i in range(edge_num + 1):
        for j in range(edge_num + 1):
            for k in range(edge_num + 1):
                if count >= N or (i == j and j == k):
                    continue
                cmap[count] = [int(step_size * n) for n in [i, j, k]]
                count += 1
    """

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    label_tensor = Colorize(n_label)(label_tensor)
    return label_tensor.float() / 255.


def numpy2label(label_np, n_label):
    img_t = Colorize(n_label)(torch.from_numpy(label_np))
    return img_t.numpy().transpose(1, 2, 0)


######
## Image processing (PIL & Numpy based)
######


def imread(fpath):
    with open(os.path.join(fpath), "rb") as f:
        return np.asarray(Image.open(f), dtype="uint8")


def imwrite(fpath, image):
    """
    image: np array, value range in [0, 255].
    """
    if ".jpg" in fpath or ".jpeg" in fpath:
        ext = "JPEG"
    elif ".png" in fpath:
        ext = "PNG"
    with open(os.path.join(fpath), "wb") as f:
        Image.fromarray(image.astype("uint8")).save(f, format=ext)


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img
    

def normalize_image(img):
    img[img < -1] = -1
    img[img > 1] = 1
    return (img+1)/2


def set_lerp_val(progression, lerp_val):
    for p in progression:
        p.lerp = lerp_val


def get_generator_blockconv_lr(g, lr):
    dic = []
    for i, blk in enumerate(g.progression):
        dic.append({"params": blk.conv1.parameters(), "lr": lr})
        dic.append({"params": blk.conv2.parameters(), "lr": lr})
        dic.append({"params": blk.noise1.parameters(), "lr": lr})
        dic.append({"params": blk.noise2.parameters(), "lr": lr})
    return dic


def get_generator_extractor_lr(g, lr):
    dic = []
    for i, blk in enumerate(g.progression):
        if "conv" in blk.segcfg:
            dic.append({"params": blk.extractor.parameters(), "lr": lr})
    return dic


######
## Others
######


def parse_noise(vec):
    """
    StyleGAN only
    """
    noise = []
    prev = 0
    for i in range(18):
        size = 4 * 2 ** (i // 2)
        noise.append(vec[prev : prev + size ** 2].view(1, 1, size, size))
        prev += size ** 2
    return noise


def list_collect_data(data_dir, keys=["origin_latent", "origin_noise", "image_stroke", "image_mask", "label_stroke", "label_mask"]):
    dic = {}
    for key in keys:
        keyfiles = glob.glob(f"{data_dir}/*{key}*")
        keyfiles.sort()
        dic[key] = keyfiles
    return dic


def onehot(x, n):
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)


def onehot_logit(x, n):
    x = x.argmax(1, keepdim=True)
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def infinite_dataloader(dl, total):
	i = 0
	while True:
		for sample in dl:
			i += 1
			if i == total:
				return
			yield sample
		dl.dataset.reset()


def set_seed(seed):
    random.seed(seed)
    #print('setting random-seed to {}'.format(seed))
    np.random.seed(seed)
    #print('setting np-random-seed to {}'.format(seed))
    torch.backends.cudnn.enabled = False
    #print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
    # set seed for CPU
    torch.manual_seed(seed)
    #print('setting torch-seed to {}'.format(seed))


def torch2numpy(x):
    try:
        return x.data.cpu().numpy()
    except:
        return x


def lerp(a, b, x, y, i):
    """
    Args:
        input from [a, b], output to [x, y], current position i
    """
    return (i - a) / (b - a) * (y - x) + x


class PLComposite(object):
    """
    Piecewise linear composition.
    """

    def __init__(self, st_x=0, st_y=0):
        super(PLComposite, self).__init__()
        self.ins = [st_x]
        self.outs = [st_y]

    # px should be sorted (add in sequential order)
    def add(self, px, py):
        self.ins.append(px)
        self.outs.append(py)

    def __call__(self, x):
        for i in range(1, len(self.ins)):
            if self.ins[i-1] <= x and x <= self.ins[i]:
                return lerp(self.ins[i-1], self.ins[i], self.outs[i-1], self.outs[i], x)


#########
### Evaluation
#########


def compute_score(y_pred, y_true):
    n_true = y_true.astype("float32").sum()
    n_pred = y_pred.astype("float32").sum()
    tp = (y_true & y_pred).astype("float32").sum()
    fp = n_pred - tp
    fn = n_true - tp
    return tp, fp, fn

def idmap(x):
    id2cid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5, 8: 6, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15}
    for fr,to in id2cid.items():
        if fr == to:
            continue
        x[x == fr] = to
    return x


def diff_idmap(x):
    """
    combine neural network output
    (N, C2, H, W) -> (N, C1, H, W)
    """
    cid2id = {0: 0, 1: 1, 2: 2, 3: 3, 4: [4, 5], 5: [6, 7], 6: [8, 9], 7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18}
    px = F.softmax(x, dim=1)
    ts = []
    for dst, src in cid2id.items():
        if type(src) is int:
            ts.append(px[:, src:src+1])
        else:
            composition = sum([px[:, index:index+1] for index in src])
            ts.append(composition)
    ps = torch.cat(ts, dim=1)
    ps /= ps.sum(dim=1, keepdim=True)
    return torch.log(ps)


class MaskCelebAEval(object):
    def __init__(self, resdic=None, map_id=True):
        self.dic = {}
        self.raw_label = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        self.dic["class"] = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        self.n_class = len(self.dic["class"])
        self.ignore_classes = [0]
        self.dic["result"] = []
        self.dic["class_result"] = [[] for i in range(self.n_class)]
        self.id_to_contiguous_id()
        self.map_id = map_id

    def id_to_contiguous_id(self):
        self.id2cid = {}
        for id, name in enumerate(self.raw_label):
            if name == "l_eye" or name == "r_eye":
                name = "eye"
            if name == "l_brow" or name == "r_brow":
                name = "brow"
            if name == "l_ear" or name == "r_ear":
                name = "ear"
            self.id2cid[id] = self.dic["class"].index(name)

    def idmap(self, x):
        for fr,to in self.id2cid.items():
            if fr == to:
                continue
            x[x == fr] = to
        return x

    def compute_score(self, seg, label):
        metrics = []
        pixelcorrect = pixeltotal = 0
        for i in range(self.n_class):
            if i in self.ignore_classes:
                precision, recall, iou = -1, -1, -1
            else:
                tp, fp, fn = compute_score(seg == i, label == i)
                pixelcorrect += tp
                pixeltotal += tp + fn
                gt_nonempty = (tp + fn) > 0
                if gt_nonempty:
                    if tp + fp == 0:
                        precision = 0
                    else:
                        precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    iou = tp / (tp + fp + fn)
                else:
                    # doesn't count if gt is empty
                    if fp > 0:
                        precision = 0
                    else:
                        precision = -1
                    recall = -1
                    iou = -1

            metrics.append([precision, recall, iou])

        pixelacc = float(pixelcorrect) / pixeltotal
        return pixelacc, metrics

    def accumulate(self, scores):
        pixelacc, metrics = scores
        self.dic["result"].append(pixelacc)
        for i, s in enumerate(metrics):
            self.dic["class_result"][i].append(s)

    def aggregate(self):
        arr = self.dic["result"]
        self.dic["pixelacc"] = float(sum(arr)) / len(arr)
        self.dic["AP"] = [-1] * self.n_class
        self.dic["AR"] = [-1] * self.n_class
        self.dic["IoU"] = [-1] * self.n_class
        for i in range(self.n_class):
            metrics = np.array(self.dic["class_result"][i])
            for j, name in enumerate(["AP", "AR", "IoU"]):
                arr = metrics[:, j]
                arr = arr[arr > -1]
                if arr.shape[0] == 0:
                    self.dic[name][i] = -1
                else:
                    self.dic[name][i] = arr.mean()
        
        for j, name in enumerate(["AP", "AR", "IoU"]):
            vals = [self.dic[name][i] for i in range(self.n_class)]
            vals = np.array(vals)
            self.dic["m" + name] = vals[vals > -1].mean()

    def summarize(self):
        print("=> mAP  \t  mAR  \t  mIoU  \t  PixelAcc")
        print("=> %.3f\t%.3f\t%.3f\t%.3f" % (self.dic["mAP"], self.dic["mAR"], self.dic["mIoU"], self.dic["pixelacc"]))
        print("=> Class wise metrics:")
        
        self.clean_dic = {}
        for key in ["mAP", "mAR", "mIoU", "pixelacc"]:
            self.clean_dic[key] = self.dic[key]

        print("=> Name \t  AP \t  AR \t  IoU \t")
        for i in range(self.n_class):
            print("=> %s: \t%.3f\t%.3f\t%.3f" % (
                self.dic["class"][i],
                self.dic["AP"][i],
                self.dic["AR"][i],
                self.dic["IoU"][i]))
            for key in ["AP", "AR", "IoU"]:
                self.clean_dic[key] = self.dic[key]
        return self.clean_dic

    def save(self, fpath):
        np.save(fpath, self.dic)

#########
## Logging related functions
#########

def str_num(n):
    return ("%.3f" % n).replace(".000", "")


def str_latex_table(strs):
    for i in range(len(strs)):
        for j in range(len(strs[0])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("_", "\\_")

    ncols = len(strs[0])
    seps = "".join(["c" for i in range(ncols)])
    s = []
    s.append("\\begin{tabular}{%s}" % seps)
    s.append(" & ".join(strs[0]) + " \\\\\\hline")
    for line in strs[1:]:
        s.append(" & ".join(line) + " \\\\")
    s.append("\\end{tabular}")

    for i in range(len(strs)):
        for j in range(len(strs[0])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("\\_", "_")

    return "\n".join(s)


def str_csv_table(strs):
    s = []
    for i in range(len(strs)):
        s.append(",".join(strs[i]))
    return "\n".join(s)


def format_agreement_result(dic):
    label_list = ['skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    global_metrics = ["pixelacc", "mAP", "mAR", "mIoU"]
    class_metrics = ["AP", "AR", "IoU"]

    n_model = len(dic["mIoU"])
    iters = [i * 1000 for i in range(1, 1 + n_model)]

    # table 1: model iterations and global accuracy
    numbers = [iters, dic["pixelacc"], dic["mAP"], dic["mAR"], dic["mIoU"]]
    numbers = np.array(numbers).transpose() # (10, 5)
    strs = [["step"] + global_metrics]
    for i in range(n_model):
        strs.append([str_num(n) for n in numbers[i]])
    # print latex table
    print(str_latex_table(strs))
    print(str_csv_table(strs))

    # table 2: classwise accuracy
    best_ind = np.argmax(dic["mIoU"])
    strs = [["metric"] + label_list]
    numbers = []
    for metric in class_metrics:
        data = dic[metric][best_ind][1:] # ignore background
        numbers.append(data)
    numbers = np.array(numbers) # (3, 16)
    for i in range(len(class_metrics)):
        strs.append(["%.3f" % n if n > -1 else "-" for n in numbers[i]])
    for i in range(1, len(strs)):
        strs[i] = [class_metrics[i - 1]] + strs[i]
    # print latex table
    print(str_latex_table(strs))
    print(str_csv_table(strs))


def format_test_result(dic):
    label_list = ['skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    global_metrics = ["pixelacc", "mAP", "mAR", "mIoU"]
    class_metrics = ["AP", "AR", "IoU"]
    
    # table 1: global metrics
    numbers = [dic[m] for m in global_metrics]
    numbers = np.array(numbers)
    strs = [global_metrics]
    strs.append([str_num(n) for n in numbers])
    # print latex table
    print(str_latex_table(strs))
    print(str_csv_table(strs))

    # table 2: classwise accuracy
    strs = [["metric"] + label_list]
    numbers = []
    for metric in class_metrics:
        data = dic[metric][1:] # ignore background
        numbers.append(data)
    numbers = np.array(numbers) # (3, 16)
    for i in range(len(class_metrics)):
        strs.append(["%.3f" % n if n > -1 else "-" for n in numbers[i]])
    for i in range(1, len(strs)):
        strs[i] = [class_metrics[i - 1]] + strs[i]
    # print latex table
    print(str_latex_table(strs))
    print(str_csv_table(strs))


def plot_dic(dic, title="", file=None):
    n = len(dic.items())
    fig = plt.figure(figsize=(3 * n, 3))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(1, n, i + 1)
        ax.plot(v)
        ax.legend([k])
    if len(title) > 0:
        plt.suptitle(title)
    if file is not None:
        plt.savefig(file)
        plt.close()
        

def parse_log(logfile):
    with open(logfile) as f:
        head = f.readline().strip().split(" ")
        dic = {h: [] for h in head}
        lines = f.readlines()

    for l in lines:
        items = l.strip().split(" ")
        for h, v in zip(head, items):
            dic[h].append(float(v))
    return dic


"""
Args:
    record: loss dic, must have loss component
"""
def write_log(expr_dir, record):
    with open(expr_dir + "/log.txt", "w") as f:
        anykey = None
        for key in record.keys():
            if anykey is None:
                anykey = key
            f.write("%s " % key)
        f.write("\n")
        for i in range(len(record[anykey])):
            for key in record.keys():
                try:
                    f.write("%f " % record[key][i])
                except:
                    print("!> No enough data at %s[%d]" % (key, i))
            f.write("\n")






""" Deprecated
def permute_masks(masks):
    def permute_(t):
        tmp = t[0]
        t[:-1] = t[1:]
        t[-1] = tmp
        return t

    for m in masks:
        if m is None:
            continue
        for i in range(len(m[0])):
            m[0][i] = permute_(m[0][i])
            m[1][i] = permute_(m[1][i])


def get_masks(blocks, step=6):
    masks = []
    for i, blk in enumerate(blocks):
        if blk.att > 0 and hasattr(blk.attention1, "mask"):
            masks.append([blk.attention1.mask, blk.attention2.mask])
        else:
            masks.append(None)
    return masks


def get_segmentation(blocks, step=-1, detach=True):
    seg = []
    for i, blk in enumerate(blocks):
        if step > 0 and step != i:
            continue
        if blk.n_class > 0:
            input = blk.seg_input
            if detach:
                input = input.detach()
            net_device = next(blk.extractor.parameters()).device
            if input.device != net_device:
                input = input.cpu().to(net_device)
            seg.append(blk.extractor(input))
    return seg

def visualize_masks(masks):
    masks_ = []
    for m in masks:
        if m is not None:
            masks_.extend(m[0])
            masks_.extend(m[1])
    masks_ = torch.cat([F.interpolate(m, 128) for m in masks_], 0)
    return masks_
"""
