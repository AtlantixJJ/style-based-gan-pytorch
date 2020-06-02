import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import model, utils, loss
from model.semantic_extractor import LinearSemanticExtractor


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


def iou(y_pred, y_true):
    tp, fp, fn = compute_score_numpy(y_pred, y_true)
    return compute_all_metric(tp, fp, fn)[-1]


class SimpleIoUMetric(object):
    def __init__(self, ignore_classes=[0], n_class=15):
        self.num = 0
        self.n_class = n_class
        self.ignore_classes = ignore_classes
        self.result = {}
        self.reset()
    
    def reset(self):
        self.num = 0
        self.global_result = {}
        self.class_result = {}
        self.class_result["IoU"] = [-1] * self.n_class
        self.result = {}
        self.result["pixelacc"] = []
        self.result["IoU"] = [[] for _ in range(self.n_class)]

    def aggregate_process(self, winsize=100):
        arr = np.array(self.result["pixelacc"])
        mask = arr > -1
        res = np.array(self.result["IoU"])
        windowsum = utils.window_sum(arr, size=winsize)
        divider = utils.window_sum(mask.astype("float32"), size=winsize)
        global_result = {"pixelacc" : windowsum / divider}
        class_result = {}
        for i in range(self.n_class):
            name = utils.CELEBA_CATEGORY[i]
            mask = res[i] > -1
            class_result[name] = res[i].copy()
            class_result[name][~mask] = 0
            if class_result[name].shape[0] > 0:
                windowsum = utils.window_sum(class_result[name], size=winsize)
                divider = utils.window_sum(mask.astype("float32"), size=winsize)
                divider[divider < 1e-5] = 1e-5
                class_result[name] = windowsum / divider
        return global_result, class_result

    def aggregate(self, start=0):
        # pixel acc
        arr = np.array(self.result["pixelacc"][start:])
        tmp = arr[arr > -1]
        self.global_result["pixelacc"] = tmp.mean() if len(tmp) > 0 else -1

        # convert to numpy array
        res = np.array(self.result["IoU"])[:, start:]

        # average over samples for each class and metric
        for i in range(self.n_class):
            arr = res[i]
            arr = arr[arr > -1]
            if arr.shape[0] == 0:
                self.class_result["IoU"][i] = -1
            else:
                self.class_result["IoU"][i] = arr.mean()
    
        # calculate global metric
        vals = np.array(self.class_result["IoU"]) # get rid of invalid classes
        tmp = vals[vals > -1]
        self.global_result[f"mIoU"] = tmp.mean() if len(tmp) > 0 else -1

    def __str__(self):
        strs = []
        strs.append("=> pixelacc: %.3f\tmIoU: %.3f" % 
            (self.global_result["pixelacc"], self.global_result["mIoU"]))
        return "\n".join(strs)

    def __call__(self, y_pred, y_true):
        self.num += 1
        c_pred = np.unique(y_pred)
        c_true = np.unique(y_true)
        iou = 0
        for i in range(self.n_class):
            if i in self.ignore_classes or (i not in c_pred and i not in c_true):
                iou = -1
            else:
                tp, fp, fn = compute_score_numpy(y_pred == i, y_true == i)
                iou = tp / (tp + fp + fn)
            self.result["IoU"][i].append(iou)

        mask = (y_pred > 0) & (y_true > 0)
        correct_mask = (y_pred[mask] == y_true[mask])
        total = mask.sum()
        pixelacc = -1
        if total > 0:
            pixelacc = correct_mask.sum() / float(total)
        self.result["pixelacc"].append(pixelacc)


class DetectionMetric(object):
    """
    Manage common detection evaluation metric
    """
    def __init__(self, ignore_classes=[0], n_class=16):
        self.num = 0
        self.n_class = n_class
        self.ignore_classes = ignore_classes
        self.class_metric_names = ["AP", "AR", "IoU"]
        self.result = {}
        self.reset()

    def load_from_dic(self, dic):
        self.result = dic

    def reset(self):
        self.num = 0
        del self.result
        self.result = {}
        self.global_result = {}
        self.class_result = {}
        self.result["pixelacc"] = []
        for name in self.class_metric_names:
            self.result[name] = [[] for _ in range(self.n_class)]
            self.class_result[name] = [-1] * self.n_class

    def aggregate(self, start=0, threshold=0):
        # pixel acc
        arr = np.array(self.result["pixelacc"])[start:]
        v = arr[arr > -1]
        self.global_result["pixelacc"] = v.mean() if len(v) > 0 else -1

        # average over samples for each class and metric
        for i in range(self.n_class):
            for j, name in enumerate(self.class_metric_names):
                arr = np.array(self.result[name][i])[start:]
                arr = arr[arr > -1]
                if arr.shape[0] == 0:
                    self.class_result[name][i] = -1
                else:
                    self.class_result[name][i] = arr.mean()
        
        # calculate global metric
        for j, name in enumerate(self.class_metric_names):
            vals = np.array(self.class_result[name]) # get rid of invalid classes
            v = vals[vals > threshold]
            self.global_result[f"m{name}"] = v.mean() if len(v) > 0 else -1
        
        return self.global_result, self.class_result

    # aggregate the result of given classes
    # need to be called after aggregate()
    def subset_aggregate(self, name, indice):
        for mname in self.class_metric_names:
            vals = np.array(self.result[mname])[indice]
            self.result[f"m{mname}_{name}"] = vals[vals > -1].mean()
            
    def __call__(self, y_pred, y_true):
        self.num += 1
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
        pixelacc = float(pixelcorrect) / pixeltotal if pixeltotal > 0 else -1

        self.result["pixelacc"].append(pixelacc)
        for i, s in enumerate(metrics): # i-th class, score
            for j, name in enumerate(self.class_metric_names): # j-th metric
                self.result[name][i].append(s[j])

        return pixelacc, metrics

    def __str__(self):
        strs = []
        strs.append(f"=> Detection metric result on {self.num} results")
        strs.append("=> pixelacc    \t  mAP  \t  mAR  \t  mIoU")
        res = [self.global_result["pixelacc"]]
        res.extend([self.global_result[f"m{name}"]
            for name in self.class_metric_names])
        strs.append("=> %.3f\t%.3f\t%.3f\t%.3f" % tuple(res))
        return "\n".join(strs)


class MaskCelebAEval(object):
    def __init__(self):
        self.face_indice = [1, 2, 4, 5, 6, 7, 8, 9, 10, 13]
        self.other_indice = [3, 11, 12, 14]
        
        self.dic = {}
        self.dic["class"] = utils.CELEBA_CATEGORY # class information
        self.ignore_classes = [0, 13]
        self.dic["result"] = [] # result of each example
        self.n_class = len(self.dic["class"])
        self.dic["class_result"] = [[] for i in range(self.n_class)]
        self.metric = DetectionMetric([0], self.n_class)
        #self.mapid = utils.CelebAIDMap()

    def calc_single(self, seg, label):
        pixelacc, metrics = self.metric(seg, label)
        return pixelacc, metrics

    def aggregate_process(self, winsize=100):
        n_class = len(self.dic['class'])
        number = len(self.dic["class_result"][0])
        global_dic = {}
        class_dic = {}
        # [BUG] this should be window sum
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
        for key in ["mAP", "mAR", "mIoU"]:
            for t in ["face", "other"]:
                self.clean_dic[f"{key}_{t}"] = self.global_result[f"{key}_{t}"]

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


class SeparabilityEvaluator(object):
    def __init__(self, model, sep_model, external_model,
            latent_dim=512, test_size=256, n_class=16, test_dl=None):
        self.device = "cuda"
        self.model = model
        self.sep_model = sep_model
        self.external_model = external_model
        self.latent_dim = latent_dim
        self.test_size = test_size
        self.n_class = n_class
        self.metric = DetectionMetric(n_class=n_class)
        self.test_dl = test_dl
        self.fix_latents = torch.from_numpy(
            np.load("datasets/simple_latent.npy"))
        self.fix_latents = self.fix_latents.float().to(self.device)
        op = getattr(self.model, "generate_noise", None)
        if callable(op):
            self.has_noise = True
            self.fix_noises = [op() for _ in range(test_size)]
        else:
            self.has_noise = False
        self.reset()

    def reset(self):
        self.global_dic = {"pixelacc":[],"mAP":[],"mAR":[],"mIoU":[]}
        self.class_dic = {k:[[] for _ in range(self.n_class)]
            for k in self.metric.class_metric_names}
        self.prev_segm = 0
        self.metric.reset()

    def __call__(self):
        segms = []
        for i in range(self.fix_latents.shape[0]):
            latent = self.fix_latents[i:i+1].to(self.device)
            if self.has_noise:
                noise = [n.to(self.device) for n in self.fix_noises[i]]
                self.model.set_noise(noise)
            image, stage = self.model.get_stage(latent, detach=True)
            seg = self.sep_model(stage)[-1]
            segms.append(seg.argmax(1))
        segms = utils.torch2numpy(torch.cat(segms))
        if self.prev_segm is 0:
            self.prev_segm = segms
            return 0

        for seg, label in zip(segms, self.prev_segm):
            self.metric(seg, label)

        global_dic, class_dic = self.metric.aggregate()
        for k in global_dic.keys():
            self.global_dic[k].append(global_dic[k])
        for k in class_dic.keys():
            for i in range(self.n_class):
                self.class_dic[k][i].append(class_dic[k][i])

        self.metric.reset()
        self.prev_segm = segms


class LinearityEvaluator(object):
    """
    External model: a semantic segmentation network
    """
    def __init__(self, model, external_model,
            last_only=1,
            train_iter=1000, batch_size=1, latent_dim=512,
            test_dl=None, test_size=256, n_class=15):
        self.model = model
        self.external_model = external_model
        self.last_only = last_only
        self.device = "cuda"
        self.train_iter = train_iter
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        latent = torch.randn(1, latent_dim, device=self.device)
        image, stage = self.model.get_stage(latent, detach=True)
        self.dims = [s.shape[1] for s in stage]
        self.sep_model = LinearSemanticExtractor(
            n_class=n_class,
            dims=self.dims).to(self.device)
        self.sep_eval = SeparabilityEvaluator(
            self.model, self.sep_model, self.external_model,
            latent_dim, test_size, n_class, test_dl)


    def __call__(self, model, name):
        self.model = model
        latent = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        for ind in tqdm(range(self.train_iter)):
            latent.normal_()
            image, stage = self.model.get_stage(latent)
            prob = self.external_model(image.clamp(-1, 1))
            segs = self.sep_model(stage, last_only=self.last_only)
            segloss = loss.segloss(segs, prob.argmax(1))
            segloss.backward()

            self.sep_model.optim.step()
            self.sep_model.optim.zero_grad()
            self.sep_eval()

            if ind % 100 == 0 or ind + 1 == self.train_iter:
                np.save(f"results/{name}_global_dic.npy", self.sep_eval.global_dic)
                np.save(f"results/{name}_class_dic.npy", self.sep_eval.class_dic)
                v = np.array(self.sep_eval.global_dic["mIoU"])
        return np.abs(v[1:] - v[:-1]).mean()
