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
THRESHOLD = 0.01
data_dir = "record/lsun"
files = glob.glob(f"{data_dir}/*.npy")
files.sort()
cg = [[0, 336], [336, 361], [361, 390]]
object_metric = evaluate.DetectionMetric(
    n_class=cg[0][1] - cg[0][0])
material_metric = evaluate.DetectionMetric(
    n_class=cg[1][1] - cg[1][0])

label_list = open("figure/lsun.txt", "r").read().split("\n")
label_list = [l.split(" ") for l in label_list]
object_list = [" ".join(l[:-1]) for l in label_list
    if l[-1] == "object"]
material_list = [" ".join(l[:-1]) for l in label_list
    if l[-1] == "material"]
np_obj_list = np.array(object_list)
np_mat_list = np.array(material_list)

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


def get_topk_classes(dic, namelist):
    res = {}
    for metric_name in ["AP", "AR", "IoU"]:
        x = np.array(dic[metric_name])
        y = x.argsort()
        k = 0
        while x[y[k]] < THRESHOLD:
            k += 1
        y = y[k:]
        y.sort()
        names = namelist[y] 
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

    def process_result(self, res, name):
        global_csv, class_csv = res[2:]
        self.global_table.append(get_row(global_csv, 0, name))
        self.class_table.append(get_row(class_csv, 0, name))
        global_latex, class_latex = res[:2]
        self.global_tabular.append(get_row(global_latex, 1, name, " & "))
        self.class_tabular.append(get_row(class_latex, 1, name, " & "))

        self.global_csv_head = global_csv.split("\n")[0]
        self.class_csv_head = class_csv.split("\n")[0]
        self.global_latex_head = global_latex.split("\n")[:2]
        self.class_latex_head = class_latex.split("\n")[:2]

    def write_class(self, subfix="object"):
        self.ct = [self.class_csv_head] + self.class_table
        self.cu = self.class_latex_head + self.class_tabular
        with open(f"{subfix}_class_result.csv", "w") as f:
            f.write("\n".join(self.ct))
        with open(f"{subfix}_class_tabular.tex", "w") as f:
            f.write("\n".join(self.cu))

    def write_global(self, subfix="object"):
        self.gt = [self.global_csv_head] + self.global_table
        self.gu = self.global_latex_head + self.global_tabular
        with open(f"{subfix}_global_result.csv", "w") as f:
            f.write("\n".join(self.gt))
        with open(f"{subfix}_global_tabular.tex", "w") as f:
            f.write("\n".join(self.gu))

object_summary = Summary()
material_summary = Summary()

# get all classes first
all_objects = []
all_materials = []
for f in files:
    name = get_name(f)
    object_dic, material_dic = np.load(f, allow_pickle=True)[:2]
    object_metric.result = object_dic
    material_metric.result = material_dic
    object_metric.aggregate(threshold=THRESHOLD)
    material_metric.aggregate(threshold=THRESHOLD)

    all_objects.append(set(get_topk_classes(
        object_metric.class_result, np_obj_list)[1]))
    all_materials.append(set(get_topk_classes(
        material_metric.class_result, np_mat_list)[1]))

all_objects = list(set.union(*all_objects))#list(set.intersection(*all_objects))
all_materials = list(set.union(*all_materials))#list(set.intersection(*all_materials))
obj_inds = np.array([object_list.index(n) for n in all_objects])
mat_inds = np.array([material_list.index(n) for n in all_materials])
print(all_objects)
print(all_materials)
for f in files:
    name = get_name(f)
    object_dic, material_dic = np.load(f, allow_pickle=True)[:2]
    object_metric.result = object_dic
    material_metric.result = material_dic
    #object_metric.aggregate(threshold=THRESHOLD)
    #material_metric.aggregate(threshold=THRESHOLD)
    object_metric.subset_aggregate("common", obj_inds)
    material_metric.subset_aggregate("common", mat_inds)

    object_dic = {
        "mIoU_common" : object_metric.result["mIoU_common"],
        "IoU" : [object_metric.class_result["IoU"][i] for i in obj_inds]}
    material_dic = {
        "mIoU_common" : material_metric.result["mIoU_common"],
        "IoU" : [material_metric.class_result["IoU"][i] for i in mat_inds]}
    material_dic["IoU"] = [v if v > 0 else 0 for v in material_dic["IoU"]]
    object_dic["IoU"] = [v if v > 0 else 0 for v in object_dic["IoU"]]
    res = utils.format_test_result(object_dic,
        global_metrics=["mIoU_common"],
        class_metrics=["IoU"],
        label_list=all_objects)
    object_summary.process_result(res, name)
    #object_summary.write_class(f"{name}_object")
    #object_summary.reset()
    
    res = utils.format_test_result(material_dic,
        global_metrics=["mIoU_common"],
        class_metrics=["IoU"],
        label_list=all_materials)
    material_summary.process_result(res, name)
    #material_summary.write_class(f"{name}_material")
    #material_summary.reset()

    """
    plt.bar(objects, object_dic['IoU'])
    plt.savefig(f"{name}_objects_agreement.png")
    plt.close()
    plt.bar(materials, material_dic['IoU'])
    plt.savefig(f"{name}_materials_agreement.png")
    plt.close()
    """

object_summary.write_class(f"object")
material_summary.write_class(f"material")
object_summary.write_global("object")
material_summary.write_global("material")