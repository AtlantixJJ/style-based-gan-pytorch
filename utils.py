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
from tqdm import tqdm
try:
    import cv2
except:
    print("!> No opencv2")


CELEBA_COLORS = [(0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),(128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(64, 128, 128),(192, 128, 128)]

CELEBA_CATEGORY = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck', 'cloth']

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


def bu(x, s):
    if type(x) is list:
        return [F.interpolate(img,
            size=s, mode="bilinear", align_corners=True) for img in x]
    elif isinstance(x, torch.Tensor):
        return F.interpolate(x,
            size=s, mode="bilinear", align_corners=True)


# make 2-exponential pyramids: (4, 8, 16, ..., 1024)
# [(N, 3, H, W)]
def make_pyramid(images):
    img1_p = torch.cat(bu(images[:4], 128))
    img1 = vutils.make_grid(img1_p, nrow=2, padding=0).unsqueeze(0)
    img2_p = torch.cat([img1] + bu(images[4:7], 256))
    img2 = vutils.make_grid(img2_p, nrow=2, padding=0).unsqueeze(0)
    return torch.cat([img2] + bu(images[7:], 512), dim=3)


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


def catlist(tensor_list, size=256):
    for i in range(len(tensor_list)):
        if len(tensor_list[i].shape) == 3:
            tensor_list[i] = tensor_list[i].unsqueeze(0)
    a = [F.interpolate(t, size=size, mode="nearest", align_corners=True)
        for t in tensor_list]
    return torch.cat(a)


def simple_dilate(img, s=1):
    res = torch.zeros_like(img)
    xs, ys = np.where(img.cpu().numpy())

    def inbox(x, y):
        return (x >= 0) and (y >= 0) and \
            (x < img.shape[0]) and (y < img.shape[1])

    def mark(x, y):
        for i in range(-s, s + 1, 1):
            for j in range(-s, s + 1, 1):
                if inbox(x + i, y + j) and img[x + i, y + j]:
                    res[x, y] = True
                    return x + i, y + i
        return False

    for x, y in zip(xs, ys):
        if res[x, y]:
            continue

        flag = mark(x, y)

        if flag:
            x, y = flag
            stx = max(0, x - s)
            edx = min(img.shape[0], x + s + 1)
            sty = max(0, y - s)
            edy = min(img.shape[1], y + s + 1)
            res[stx:edx, sty:edy] = True

    return res

    
POSITIVE_COLOR = cm.get_cmap("Reds")
NEGATIVE_COLOR = cm.get_cmap("Blues")
def heatmap_numpy(image):
    """
    assume numpy array as input: (N, H, W) in [0, 1]
    returns: (N, H, W, 3)
    """
    image1 = image.copy()
    mask1 = image1 > 0
    image1[~mask1] = 0

    image2 = -image.copy()
    mask2 = image2 > 0
    image2[~mask2] = 0

    pos_img = POSITIVE_COLOR(image1)[:, :, :, :3]
    neg_img = NEGATIVE_COLOR(image2)[:, :, :, :3]

    x = np.ones_like(pos_img)
    x[mask1] = pos_img[mask1]
    x[mask2] = neg_img[mask2]

    return x


def heatmap_torch(tensor):
    """
    assume 4D torch.Tensor (N, 1, H, W)
    """
    numpy_arr = torch2numpy(tensor[:, 0, :, :])
    heatmap = heatmap_numpy(numpy_arr)
    t = torch.from_numpy(heatmap.transpose(0, 3, 1, 2)).float()
    return t.to(tensor.device)


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)

    def colorize_single(self, gray_image):
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

    def __call__(self, images):
        if (len(images.shape) == 4 and images.shape[1] == 1) or len(images.shape) == 3:
            colorization = [self.colorize_single(img) for img in images]
            if isinstance(images, torch.Tensor):
                return torch.stack(colorization)
            else:
                return np.stack(colorization)
        else:
            return self.colorize_single(images)



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


def get_group(labels, bg=True):
    prev_cat = labels[0][1]
    prev_idx = 0
    cur_idx = 0
    groups = []

    for label, cat in labels:
        if cat != prev_cat:
            if bg:
                cur_idx += 1 # plus one for unlabeled class
            groups.append([prev_idx, cur_idx])
            prev_cat = cat
            prev_idx = cur_idx
        cur_idx += 1
    groups.append([prev_idx, cur_idx + 1])
    return groups
    

def resolution_from_name(fpath):
    resolution = 1024
    if "256x256" in fpath:
        resolution = 256
    return resolution



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
"""
def get_region(stroke_mask, label):
    region_map, n_region = random_integrated_floodfill(label.copy())
    l = np.bincount(region_map[stroke_mask, 0]).argmax()
    return region_map[:, :, 0] == l
"""


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


def rgb2label(image, color_list):
    t = torch.zeros(image.shape[1:]).float()
    for i, c in enumerate(color_list):
        t[color_mask_tensor(image, c)] = i
    return t


class Timer(object):    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


# map given id and reduce other id to form continuous ids
# from > to, both refer to index in original labeling
def create_id2cid(n, map_from=[2], map_to=[1]):
    mid2id = {i:[i] for i in range(n)}
    for fr, to in zip(map_from, map_to):
        ind = mid2id[fr].index(fr)
        del mid2id[fr][ind]
        mid2id[to].append(fr)

    id2cid = {}
    count = 0
    for i, (k, v) in enumerate(mid2id.items()):
        if len(v) == 0:
            continue
        for l in v:
            id2cid[l] = count
        count += 1

    return id2cid


def create_cid2id(n, map_from=[2], map_to=[1]):
    mid2id = {i:[i] for i in range(n)}
    for fr, to in zip(map_from, map_to):
        ind = mid2id[fr].index(fr)
        del mid2id[fr][ind]
        mid2id[to].append(fr)

    count = 0
    cid2id = {}
    count = 0
    for i, (k, v) in enumerate(mid2id.items()):
        if len(v) == 0:
            continue
        cid2id[count] = v
        count += 1

    return cid2id


def idmap(x, id2cid=None, n=None, map_from=None, map_to=None):
    if id2cid is None:
        id2cid = create_id2cid(n, map_from, map_to)
    y = 0
    if isinstance(x, torch.Tensor):
        y = torch.zeros_like(x)
    elif isinstance(x, np.ndarray):
        y = np.zeros_like(x)
    for fr,to in id2cid.items():
        y[x == fr] = to
    return y


def diff_idmap_softmax(x, cid2id=None, n=None, map_from=None, map_to=None):
    """
    combine neural network output
    (N, C2, H, W) -> (N, C1, H, W)
    """
    if cid2id is None:
        cid2id = create_cid2id(n, map_from, map_to)
    px = F.softmax(x, dim=1)
    ts = []
    for dst, src in cid2id.items():
        composition = sum([px[:, index:index+1] for index in src])
        ts.append(composition)
    ps = torch.cat(ts, dim=1)
    ps /= ps.sum(dim=1, keepdim=True)
    return ps


def diff_idmap_logit(x, cid2id=None, n=None, map_from=None, map_to=None):
    """
    combine neural network output
    (N, C2, H, W) -> (N, C1, H, W)
    """
    if cid2id is None:
        cid2id = create_cid2id(n, map_from, map_to)
    px = F.softmax(x, dim=1)
    ts = []
    for dst, src in cid2id.items():
        if len(src) == 1:
            i = src[0]
            ts.append(x[:, i:i+1])
        else:
            pr = px[:, src]
            pr = pr / pr.sum(1, keepdim=True)
            ts.append((x[:, src] * pr).sum(1, keepdim=True))
    res = torch.cat(ts, dim=1)
    return res
    

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


def listkey_convert(name, listkey, output=None):
    for i, key in enumerate(listkey):
        if key in name:
            if output is not None:
                return output[i]
            return key
    return ""


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
                strs[i][j] = strs[i][j].replace("_", "-")

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


def format_test_result(dic,
    global_metrics=["pixelacc", "mAP", "mAR", "mIoU", "mIoU_face", "mIoU_other"],
    class_metrics=["AP", "AR", "IoU"],
    label_list=CELEBA_CATEGORY):
    class_metrics = ["AP", "AR", "IoU"]

    # table 1: global metrics
    global_latex = global_csv = 0
    numbers = [0] + [dic[m] for m in global_metrics]
    numbers = np.array(numbers)
    strs = [["model"] + global_metrics]
    strs.append([str_num(n) for n in numbers])
    # print latex table
    global_latex = str_latex_table(strs)
    global_csv = str_csv_table(strs)

    # table 2: classwise accuracy
    strs = [["model", "metric"] + label_list]
    numbers = []
    for metric in class_metrics:
        data = dic[metric]
        numbers.append(data)
    numbers = np.array(numbers) # (3, 16)
    for i in range(len(class_metrics)):
        strs.append(["%.3f" % n if n > -1 else "-" for n in numbers[i]])
    for i in range(1, len(strs)):
        strs[i] = [class_metrics[i - 1]] + strs[i]
    # print latex table
    class_latex = str_latex_table(strs)
    class_csv = str_csv_table(strs)

    return global_latex, class_latex, global_csv, class_csv

def plot_dic(dic, title="", file=None):
    n = len(dic.items())
    edge = int(math.sqrt(n))
    if edge ** 2 < n:
        edge += 1
    fig = plt.figure(figsize=(4 * edge, 3 * edge))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(edge, edge, i + 1)
        if type(v[0]) is list or type(v[0]) is tuple:
            arr = np.array(v)
            ax.plot(v[:, 0], v[:, 1])
        else:
            ax.plot(v)
        ax.set_title(k)
    if len(title) > 0:
        plt.suptitle(title)
    plt.tight_layout()
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


# for celeba mask dataset
class CelebAIDMap(object):
    def __init__(self):
        self.n_class = 19
        self.map_from = [5, 7, 9]
        self.map_to = [4, 6, 8]
        self.id2cid = create_id2cid(self.n_class, self.map_from, self.map_to)
        self.cid2id = create_cid2id(self.n_class, self.map_from, self.map_to)

    def mapid(self, x):
        return idmap(x, self.id2cid)
    
    def diff_mapid(self, x):
        return diff_idmap_logit(x, self.cid2id)

    def diff_mapid_softmax(self, x):
        return diff_idmap_softmax(x, self.cid2id)
"""
