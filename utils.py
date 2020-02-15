from __future__ import print_function
import time, math, glob, os, sys
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join as osj
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from skimage import morphology
try:
    import cv2
except:
    print("!> No opencv2")


CELEBA_COLORS = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),(64, 128, 128),(192, 128, 128)]


CELEBA_FULL_CATEGORY = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

CELEBA_REDUCED_CATEGORY = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']


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
## Tensor operations
######


def torch2numpy(x):
    try:
        return x.detach().cpu().numpy()
    except:
        return x


def tensor2image(x):
    x = torch2numpy(x)[0].transpose(1, 2, 0)
    return ((x + 1) * 127.5).astype("uint8")


def lerp(a, b, x, y, i):
    """
    Args:
        input from [a, b], output to [x, y], current position i
    """
    return (i - a) / (b - a) * (y - x) + x


def onehot(x, n):
    z = torch.zeros(x.shape[0], n, x.shape[2], x.shape[3])
    return z.scatter_(1, x, 1)


def onehot_logit(x):
    label = x.argmax(1, keepdim=True)
    z = torch.zeros_like(x)
    return z.scatter_(1, label, 1)


def adaptive_sumpooling(x, size):
    h, w = x.shape[2:]
    y = F.adaptive_avg_pool2d(x, size)
    nh, nw = y.shape[2:]
    return y * float(h) * w / nh / nw


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


def tensor_resize_by_pil(x, size=(299, 299), resample=Image.BILINEAR):
    """
    x: [-1, 1] torch tensor (N, C, H, W)
    """
    y = np.zeros((x.shape[0], size[0], size[1], 3), dtype='uint8')
    x_arr = ((x + 1) * 127.5).detach().cpu().numpy().astype("uint8")
    x_arr = x_arr.transpose(0, 2, 3, 1)
    for i in range(x_arr.shape[0]):
        if x_arr.shape[-1] == 1:
            y[i] = np.asarray(Image.fromarray(x_arr[i, :, :, 0]).resize(
                size, resample).convert("RGB"))
        else:
            y[i] = np.asarray(Image.fromarray(x_arr[i]).resize(size, resample))
    return torch.from_numpy(y.transpose(0, 3, 1, 2)).type_as(x) / 127.5 - 1


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
        Image.fromarray(image.astype("uint8")).convert("RGB").save(f, format=ext)


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img
    

def imresize(image, size):
    return np.array(Image.fromarray(image).resize(size))


# clip and normalize from [-1, 1] to [0, 1]
def normalize_image(img):
    img[img < -1] = -1
    img[img > 1] = 1
    return (img+1)/2


######
## Network helper function
######


def parse_noise_stylegan(vec):
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
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)


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


def get_square_mask(mask, pad_box=0):
    w = np.where(mask.max(0))[0]
    h = np.where(mask.max(1))[0]
    xmin, xmax = w.min(), w.max() + 1
    ymin, ymax = h.min(), h.max() + 1
    cx = (xmin + xmax) / 2.; cy = (ymin + ymax) / 2.
    dw = xmax - xmin; dh = ymax - ymin
    dd = max(dw, dh)
    dd = min(mask.shape[0], dd + pad_box * 2)
    d = dd / 2.
    xmin, xmax = int(cx - d), int(cx + d)
    ymin, ymax = int(cy - d), int(cy + d)
    dx = dy = 0
    if xmin < 0:
        dx = -xmin
    elif xmax > mask.shape[1]:
        dx = mask.shape[1] - xmax # negative
    if ymin < 0:
        dy = -ymin
    elif ymax > mask.shape[0]:
        dy = mask.shape[0] - ymax # negative
    xmin += dx; xmax += dx; ymin += dy; ymax += dy
    return xmin, ymin, xmax, ymax


def image2part_catetory(image, label, n_class=16, MIN_AREA=255, pad_box=20, ignore_label=0):
    parts = []
    for i in range(n_class): # for each connected component
        if i == ignore_label:
            continue
        mask = label == i
        size = mask.sum()
        # ignore background
        # ignore image patch smaller than 15x15
        if size < MIN_AREA: 
            continue

        xmin, ymin, xmax, ymax = get_square_mask(mask, pad_box)
        
        dst_img = image[ymin:ymax, xmin:xmax].copy()
        src_img = image[ymin:ymax, xmin:xmax].copy()
        cx = dst_img.shape[1] // 2
        cy = dst_img.shape[0] // 2
        submask = ~mask[ymin:ymax, xmin:xmax]
        outside_mean = src_img[submask].mean(0).astype("uint8")
        src_img[:, :] = outside_mean

        dst_img = cv2.seamlessClone(src_img, dst_img, 255 * submask.astype("uint8"), (cx, cy), cv2.NORMAL_CLONE)
        parts.append([i, dst_img])

    # returns numpy uint8
    return parts


def image2part_connected(image, label, MIN_AREA=255, ignore_label=0):
    conn_label, conn_number = morphology.label(label, connectivity=2, return_num=True)
    parts = []
    for i in range(conn_number): # for each connected component
        mask = conn_label == i
        size = mask.sum()
        c = label[mask][0]
        # ignore background
        # ignore image patch smaller than 15x15
        if size < MIN_AREA or c == ignore_label: 
            continue

        xmin, ymin, xmax, ymax = get_square_mask(mask)
        
        dst_img = image[ymin:ymax, xmin:xmax].copy()
        src_img = image[ymin:ymax, xmin:xmax].copy()
        cx = dst_img.shape[1] // 2
        cy = dst_img.shape[0] // 2
        submask = ~mask[ymin:ymax, xmin:xmax]
        outside_mean = src_img[submask].mean(0).astype("uint8")
        src_img[:, :] = outside_mean

        dst_img = cv2.seamlessClone(src_img, dst_img, 255 * submask.astype("uint8"), (cx, cy), cv2.NORMAL_CLONE)
        parts.append([c, dst_img])

    # returns numpy uint8
    return parts


def fast_random_seed(mark, val=False, n_retry=1000):
    h, w = mark.shape
    for _ in range(n_retry):
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        if mark[x, y] == val:
            return x, y
    return -1, -1


def slow_random_seed(mark, val=False):
    xs, ys = np.where(mark == val)
    index = np.random.randint(0, len(xs))
    return xs[index], ys[index]


def random_integrated_floodfill(image):
    x = y = 0
    H, W, _ = image.shape
    label = 1 # 0 is ignored label
    mask = np.zeros((H + 2, W + 2), dtype="uint8")
    mark = np.zeros((H, W), dtype="bool")

    mode = 0 # 0: fast; 1: medium; 2: complete

    while not mark.all():
        if mode == 0: # fast random seeding
            x, y = fast_random_seed(mark)
        elif mode == 1: # strict random seeding
            x, y = slow_random_seed(mark)
        # failed to find a seed
        if mark[x, y] and mode == 0:
            mode = 1
            continue
        # flood fill
        mask.fill(0)
        number, _, _, rect = cv2.floodFill(image, mask, (y, x), label, loDiff=0, upDiff=0)
        p = [rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]]
        submask = mask[p[0]+1:p[2]+1,p[1]+1:p[3]+1].astype("bool")
        # ignore small region
        if number < 25:
            # use bounding box to reduce memory access
            image[p[0]:p[2],p[1]:p[3]][submask].fill(0)
        else:
            label += 1
        # complete markings
        mark[p[0]:p[2],p[1]:p[3]] |= submask
    return image, label


"""
Args:
    stroke_mask: the mask ([H, W] np.array bool)
    label: the visualized image ([H, W, 3] np.array uint8)
Returns:
    mask: a mask representing the region ([H, W] np.array bool)
"""
def get_region(stroke_mask, label):
    region_map, n_region = random_integrated_floodfill(label.copy())
    l = np.bincount(region_map[stroke_mask, 0]).argmax()
    return region_map[:, :, 0] == l


def color_mask(image, color):
    r = image[:, :, 0] == color[0]
    g = image[:, :, 1] == color[1]
    b = image[:, :, 2] == color[2]
    return r & g & b


def color_mask_tensor(image, color):
    r = image[0, :, :] == color[0]
    g = image[1, :, :] == color[1]
    b = image[2, :, :] == color[2]
    return r & g & b
    

def celeba_rgb2label(image):
    t = torch.zeros(image.shape[1:]).float()
    for i, c in enumerate(CELEBA_COLORS):
        t[color_mask_tensor(image, c)] = i
    return t


class Timer(object):    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def compute_score(y_pred, y_true):
    n_true = y_true.astype("float32").sum()
    n_pred = y_pred.astype("float32").sum()
    tp = (y_true & y_pred).astype("float32").sum()
    fp = n_pred - tp
    fn = n_true - tp
    return tp, fp, fn


# map given id and reduce other id to form continuous ids
def create_id2cid(n, map_from=[2], map_to=[1]):
    count = 0
    id2cid = {}
    for i in range(n):
        id2cid[]


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
        self.raw_label = CELEBA_FULL_CATEGORY
        self.dic["class"] = CELEBA_REDUCED_CATEGORY
        self.face_indice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
        self.other_indice = [11, 12, 13, 15]
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

    def aggregate_process(self, winsize=100):
        n_class = len(self.dic['class'])
        number = len(self.dic["class_result"][0])
        global_dic = {}
        class_dic = {}
        global_dic["pixelacc"] = np.cumsum(self.dic['result']) / np.arange(1, number + 1)
        class_dic["AP"] = np.zeros((n_class, number))
        class_dic["AR"] = np.zeros((n_class, number))
        class_dic["IoU"] = np.zeros((n_class, number))

        for i in range(n_class):
            metrics = np.array(self.dic["class_result"][i])
            for j, name in enumerate(["AP", "AR", "IoU"]):
                mask = metrics[:, j] > -1
                class_dic[name][i, mask] = metrics[mask, j]
                if mask.shape[0] == 0:
                    class_dic[name][i, :] = 0
                else:
                    windowsum = window_sum(class_dic[name][i, :], size=winsize)
                    divider = window_sum(mask.astype("float32"), size=winsize)
                    divider[divider < 1e-5] = 1e-5
                    class_dic[name][i, :] = windowsum / divider
        
        for j, name in enumerate(["AP", "AR", "IoU"]):
            global_dic[f"m{name}"] = class_dic[name].mean(0)
            for t in ["face", "other"]:
                arr = class_dic[name][getattr(self, f"{t}_indice"), :]
                global_dic[f"m{name}_{t}"] = arr.mean(0)
        
        self.dic["global_process"] = global_dic
        self.dic["class_process"] = class_dic
        return global_dic, class_dic

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

        for name in ["AP", "AR", "IoU"]:
            for t in ["face", "other"]:
                vals = np.array(self.dic[name])[getattr(self, f"{t}_indice")]
                self.dic[f"m{name}_{t}"] = vals[vals > -1].mean()

    def summarize(self):
        print("=> mAP  \t  mAR  \t  mIoU  \t  PixelAcc")
        print("=> %.3f\t%.3f\t%.3f\t%.3f" % (self.dic["mAP"], self.dic["mAR"], self.dic["mIoU"], self.dic["pixelacc"]))
        print("=> Face accuracy:")
        print("=> mAP  \t  mAR  \t  mIoU")
        print("=> %.3f\t%.3f\t%.3f" % (self.dic["mAP_face"], self.dic["mAR_face"], self.dic["mIoU_face"]))
        print("=> Other accuracy:")
        print("=> mAP  \t  mAR  \t  mIoU")
        print("=> %.3f\t%.3f\t%.3f" % (self.dic["mAP_other"], self.dic["mAR_other"], self.dic["mIoU_other"]))
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

    def load(self, fpath):
        self.dic = np.load(fpath, allow_pickle=True)[()]

#########
## Logging related functions
#########


"""
ORIGINAL_STDOUT = 0


def stdout_redirect(filename):
    global ORIGINAL_STDOUT
    ORIGINAL_STDOUT = sys.stdout
    sys.stdout = open(filename, "w")


# must be called after redirect
def stdout_resume():
    sys.stdout.close()
    sys.stdout = ORIGINAL_STDOUT
"""


def list_collect_data(data_dir, keys=["origin_latent", "origin_noise", "image_stroke", "image_mask", "label_stroke", "label_mask"]):
    dic = {}
    for key in keys:
        keyfiles = glob.glob(f"{data_dir}/*{key}*")
        keyfiles.sort()
        dic[key] = keyfiles
    return dic
    

def str_num(n):
    return ("%.3f" % n).replace(".000", "")


def str_latex_table(strs):
    for i in range(len(strs)):
        for j in range(len(strs[i])):
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
        for j in range(len(strs[i])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("\\_", "_")

    return "\n".join(s)


def str_csv_table(strs):
    s = []
    for i in range(len(strs)):
        s.append(",".join(strs[i]))
    return "\n".join(s)


def format_agreement_result(dic):
    label_list = CELEBA_REDUCED_CATEGORY
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

    model_global_latex = str_latex_table(strs)
    model_global_csv = str_csv_table(strs)

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
    class_latex = str_latex_table(strs)
    class_csv = str_csv_table(strs)

    return model_global_latex, class_latex, model_global_csv, class_csv, best_ind


def format_test_result(dic):
    label_list = CELEBA_REDUCED_CATEGORY
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
    edge = int(math.sqrt(n))
    if edge ** 2 < n:
        edge += 1
    fig = plt.figure(figsize=(3 * edge, 3 * edge))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(edge, edge, i + 1)
        ax.plot(v)
        ax.set_title(k)
    if len(title) > 0:
        plt.suptitle(title)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()


def plot_heatmap(dic, title="", file=None):
    n = len(dic.items())
    edge = int(math.sqrt(n))
    if edge ** 2 < n:
        edge += 1
    fig = plt.figure(figsize=(3 * edge, 3 * edge))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(edge, edge, i + 1)
        ax.imshow(v)
        ax.set_title(k)
    if len(title) > 0:
        plt.suptitle(title)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()

        
"""
Args:
    arr : 1D numpy array
"""
def window_sum(arr, size=10):
    cumsum = np.cumsum(arr)
    windowsum = np.zeros_like(cumsum)
    windowsum[:size] = cumsum[:size]
    windowsum[size:] = cumsum[size:] - cumsum[:-size]
    return windowsum


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






""" Deprecated
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
