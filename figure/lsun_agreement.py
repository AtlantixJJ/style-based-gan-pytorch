import sys, glob
sys.path.insert(0, ".")
import numpy as np
import utils, evaluate

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
THRESHOLD = 0.1
data_dir = "record/lsun"
files = glob.glob(f"{data_dir}/*.npy")
files.sort()
cg = [[0, 336], [336, 361], [361, 390]]
object_metric = evaluate.DetectionMetric(
    n_class=cg[0][1] - cg[0][0])
material_metric = evaluate.DetectionMetric(
    n_class=cg[1][1] - cg[1][0])

label_list = open("figure/lsun.txt", "r").read().split("\n")
label_list = [l.split(" ")[0] for l in label_list]
label_list = np.array(label_list)


def get_name(model_path):
    names = ["StyleGAN-Bedroom", "PGAN-Bedroom", "PGAN-Church", "StyleGAN2-Church"]
    keywords = ["bedroom_stylegan", "bedroom_proggan", "church_proggan", "church_stylegan2"]
    task = utils.listkey_convert(model_path,
        ["bedroom", "church"])
    model = utils.listkey_convert(model_path,
        ["stylegan2", "stylegan", "proggan"])
    method = utils.listkey_convert(model_path,
        ["nonlinear", "linear", "unitnorm", "unit", "generative", "spherical"])
    return f"{task}_{model}_{method}"


def get_topk_classes(dic, start=0):
    res = {}
    for metric_name in ["AP", "AR", "IoU"]:
        x = np.array(dic[metric_name])
        y = x.argsort()
        k = 0
        while x[y[k]] < THRESHOLD:
            k += 1
        y = y[k:]
        y.sort()
        # all classes are the same
        names = label_list[y + start] 
        res[metric_name] = x[y] #(names, x[y])
    return res, names.tolist()


def get_row(table, ind, name, delimiter=","):
    row = table.split("\n")[ind + 1]
    items = row.split(delimiter)
    return delimiter.join([name] + items)


class Summary(object):
    def __init__(self):
        self.gt = []
        self.ct = []
        self.gu = []
        self.cu = []
        self.reset()

    def reset(self):
        self.global_table = []
        self.class_table = []
        self.global_tabular = []
        self.class_tabular = []
        self.ct = []
        self.cu = []

    def accumulate(self):
        self.gt.extend(self.global_table)
        self.ct.extend(self.class_table)
        self.gu.extend(self.global_tabular)
        self.cu.extend(self.class_tabular)

    def process_result(self, res, name):
        global_csv, class_csv = res[2:]
        self.global_table.append(get_row(global_csv, 0, name))
        self.class_table.append(get_row(class_csv, 2, name))
        global_latex, class_latex = res[:2]
        self.global_tabular.append(get_row(global_latex, 1, name, " & "))
        self.class_tabular.append(get_row(class_latex, 3, name, " & "))

        self.global_csv_head = global_csv.split("\n")[0]
        self.class_csv_head = class_csv.split("\n")[0]
        self.global_latex_head = global_latex.split("\n")[:2]
        self.class_latex_head = class_latex.split("\n")[:2]

        self.accumulate()

    def write_class(self, subfix="object"):
        self.ct = [self.class_csv_head] + self.ct
        self.cu = self.class_latex_head + self.cu
        with open(f"{subfix}_class_result.csv", "w") as f:
            f.write("\n".join(self.ct))
        with open(f"{subfix}_class_tabular.tex", "w") as f:
            f.write("\n".join(self.cu))

    def write_global(self, subfix="object"):
        self.gt = [self.global_csv_head] + self.gt
        self.gu = self.global_latex_head + self.gu
        with open(f"{subfix}_global_result.csv", "w") as f:
            f.write("\n".join(self.gt))
        with open(f"{subfix}_global_tabular.tex", "w") as f:
            f.write("\n".join(self.gu))

object_summary = Summary()
material_summary = Summary()
for f in files:
    name = get_name(f)
    object_dic, material_dic = np.load(f, allow_pickle=True)[:2]
    object_metric.result = object_dic
    material_metric.result = material_dic
    object_metric.aggregate(threshold=THRESHOLD)
    material_metric.aggregate(threshold=THRESHOLD)

    object_dic, objects = get_topk_classes(
        object_metric.class_result, cg[0][0])
    material_dic, materials = get_topk_classes(
        material_metric.class_result, cg[1][0])
    object_dic.update(object_metric.global_result)
    material_dic.update(material_metric.global_result)

    """
    plt.bar(objects, object_dic['IoU'])
    plt.savefig(f"{name}_objects_agreement.png")
    plt.close()
    plt.bar(materials, material_dic['IoU'])
    plt.savefig(f"{name}_materials_agreement.png")
    plt.close()
    """

    res = utils.format_test_result(object_dic,
        global_metrics=["pixelacc", "mAP", "mAR", "mIoU"],
        label_list=objects)
    object_summary.process_result(res, name)
    object_summary.write_class(f"{name}_object")
    object_summary.reset()
    
    res = utils.format_test_result(material_dic,
        global_metrics=["pixelacc", "mAP", "mAR", "mIoU"],
        label_list=materials)
    material_summary.process_result(res, name)
    material_summary.write_class(f"{name}_material")
    material_summary.reset()


object_summary.write_global("object")
material_summary.write_global("material")