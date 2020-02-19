import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import model, utils, loss


def compute_score_numpy(y_pred, y_true):
    n_true = y_true.astype("float32").sum()
    n_pred = y_pred.astype("float32").sum()
    tp = (y_true & y_pred).astype("float32").sum()
    fp = n_pred - tp
    fn = n_true - tp
    return tp, fp, fn


"""
Do the computation.
"""
def compute_all_metric(tp, fp, fn):
    pixelcorrect = tp
    pixeltotal = tp + fn
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
    return pixelcorrect, pixeltotal, precision, recall, iou


class DetectionMetric(object):
    """
    Manage common detection evaluation metric
    """
    def __init__(self, ignore_classes=[0], n_class=16):
        self.n_class = n_class
        self.ignore_classes = ignore_classes
        self.class_metric_names = ["AP", "AR", "IoU"]
        self.result = {}
        self.reset()

    def reset(self):
        del self.result
        self.result = {}
        self.global_result = {}
        self.class_result = {}
        self.result["pixelacc"] = []
        for name in self.class_metric_names:
            self.result[name] = [[] for _ in range(self.n_class)]
            self.class_result[name] = [-1] * self.n_class

    def aggregate(self):
        # pixel acc
        arr = self.result["pixelacc"]
        self.global_result["pixelacc"] = float(sum(arr)) / len(arr)

        # convert to numpy array
        for name in self.class_metric_names:
            self.result[name] = np.array(self.result[name])

        # average over samples for each class and metric
        for i in range(self.n_class):
            for j, name in enumerate(self.class_metric_names):
                arr = self.result[name][i]
                arr = arr[arr > -1]
                if arr.shape[0] == 0:
                    self.class_result[name][i] = -1
                else:
                    self.class_result[name][i] = arr.mean()
        
        # calculate global metric
        for j, name in enumerate(self.class_metric_names):
            vals = self.result[name] # get rid of invalid classes
            self.global_result[f"m{name}"] = vals[vals > -1].mean()

    # aggregate the result of given classes
    # need to be called after aggregate()
    def subset_aggregate(self, name, indice):
        for mname in self.class_metric_names:
            vals = self.result[mname][indice]
            self.result[f"m{mname}_{name}"] = vals[vals > -1].mean()
            

    def __call__(self, y_pred, y_true):
        metrics = []
        pixelcorrect = pixeltotal = 0
        for i in range(self.n_class):
            if i in self.ignore_classes:
                precision, recall, iou = -1, -1, -1
            else:
                tp, fp, fn = compute_score_numpy(y_pred == i, y_true == i)
                pc, pt, precision, recall, iou = compute_all_metric(tp, fp, fn)
                pixelcorrect += pc
                pixeltotal += pt
            metrics.append([precision, recall, iou])
        pixelacc = float(pixelcorrect) / pixeltotal

        self.result["pixelacc"].append(pixelacc)
        for i, s in enumerate(metrics): # i-th class, score
            for j, name in enumerate(self.class_metric_names): # j-th metric
                self.result[name][i].append(s[j])

        return pixelacc, metrics


class MaskCelebAEval(object):
    def __init__(self):
        self.face_indice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
        self.other_indice = [11, 12, 13, 15]
        
        self.dic = {}
        self.dic["class"] = utils.CELEBA_REDUCED_CATEGORY # class information
        self.ignore_classes = [0]
        self.dic["result"] = [] # result of each example
        self.n_class = len(self.dic["class"])
        self.dic["class_result"] = [[] for i in range(self.n_class)]
        self.metric = DetectionMetric([0], self.n_class)
        self.mapid = utils.CelebAIDMap()

    def calc_single(self, seg, label):
        pixelacc, metrics = self.metric(seg, label)
        return pixelacc, metrics

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
                    windowsum = utils.window_sum(class_dic[name][i, :], size=winsize)
                    divider = utils.window_sum(mask.astype("float32"), size=winsize)
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
        self.metric.aggregate()
        self.global_result = self.metric.global_result
        self.class_result = self.metric.class_result

        for t in ["face", "other"]:
            self.metric.subset_aggregate(t, getattr(self, f"{t}_indice"))
            for name in self.metric.class_metric_names:
                k = f"m{name}_{t}"
                self.global_result[k] = self.metric.result[k]

        return self.global_result, self.class_result

    def summarize(self):
        print("=> mAP  \t  mAR  \t  mIoU")
        res = [self.global_result[f"m{name}"]
            for name in self.metric.class_metric_names]
        print("=> %.3f\t%.3f\t%.3f" % tuple(res))
        print("=> Face accuracy:")
        print("=> mAP  \t  mAR  \t  mIoU")
        res = [self.global_result[f"m{name}_face"]
            for name in self.metric.class_metric_names]
        print("=> %.3f\t%.3f\t%.3f" % tuple(res))
        print("=> Other accuracy:")
        print("=> mAP  \t  mAR  \t  mIoU")
        res = [self.global_result[f"m{name}_other"]
            for name in self.metric.class_metric_names]
        print("=> %.3f\t%.3f\t%.3f" % tuple(res))
        print("=> Class wise metrics:")
        
        self.clean_dic = {}
        for key in ["mAP", "mAR", "mIoU", "pixelacc"]:
            self.clean_dic[key] = self.global_result[key]

        print("=> Name \t  AP \t  AR \t  IoU \t")
        for i in range(self.n_class):
            print("=> %s: \t%.3f\t%.3f\t%.3f" % (
                self.dic["class"][i],
                self.class_result["AP"][i],
                self.class_result["AR"][i],
                self.class_result["IoU"][i]))
            for key in ["AP", "AR", "IoU"]:
                self.clean_dic[key] = self.class_result[key]
        return self.clean_dic

    def save(self, fpath):
        np.save(fpath, self.dic)

    def load(self, fpath):
        self.dic = np.load(fpath, allow_pickle=True)[()]


class LinearityEvaluator(object):
    """
    External model: a semantic segmentation network
    """
    def __init__(self, model, external_model,
        N=1000, imsize=512, latent_dim=512, n_class=16, stylegan_noise=False):
        self.model = model
        self.external_model = external_model
        self.device = "cuda"
        self.N = N
        self.latent_dim = latent_dim
        self.imsize = 512
        self.n_class = n_class
        self.stylegan_noise = stylegan_noise
        self.logsoftmax = torch.nn.CrossEntropyLoss()
        self.logsoftmax.to(self.device)

        self.fix_latents = torch.randn(256, self.latent_dim)
        if self.stylegan_noise:
            self.fix_noises = [self.model.generate_noise() for _ in range(256)]

        self.metric = DetectionMetric()

    def aggregate(self):
        self.metric.aggregate()

        for t in ["face", "other"]:
            self.metric.subset_aggregate(t, getattr(self, f"{t}_indice"))
        
        self.global_result = self.metric.global_result
        self.class_result = self.metric.class_result

        return self.global_result, self.class_result

    def eval_fix(self):
        segms = []
        for i in range(self.fix_latents.shape[0]):
            latent = self.fix_latents[i:i+1].detach().clone().to(self.device)
            # stylegan need a noise
            if self.stylegan_noise:
                noise = [n.detach().clone().to(self.device) for n in self.fix_noises[i]]
                self.model.set_noise(noise)
            self.model(latent)
            seg = self.extractor(self.model.stage)[-1]
            label = seg.argmax(1)
            segms.append(label.detach().clone())
        segms = utils.torch2numpy(torch.cat(segms))
        if self.prev_segm is 0:
            self.prev_segm = segms
            return 0

        for seg, label in zip(segms, self.prev_segm):
            self.metric(seg, label)

        self.prev_segm = segms

    def __call__(self, model, name):
        self.model = model
        latent = torch.randn(1, self.latent_dim, device=self.device)
        for ind in tqdm(range(self.N)):
            latent.normal_()
            image = self.model(latent)
            if ind == 0:
                self.prev_segm = 0
                self.dims = [s.shape[1] for s in self.model.stage]
                self.extractor = model.linear.LinearSemanticExtractor(self.n_class, self.dims)
            segs = self.extractor(self.model.stage)

            ext_label = self.external_model(image.clamp(-1, 1))

            segloss = loss.segloss(segs, ext_label)
            segloss.backward()

            self.extractor.optim.step()
            self.extractor.optim.zero_grad()
            self.eval_fix()

        self.metric.aggregate()
        global_dic, class_dic = self.metric.global_result, self.metric.class_result
        np.save(f"results/{name}_global_dic.npy", global_dic)
        np.save(f"results/{name}_class_dic.npy", class_dic)
        return np.array(global_dic["mIoU"]).std()
