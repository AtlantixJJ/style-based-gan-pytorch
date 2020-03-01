import sys, glob
sys.path.insert(0, ".")
import numpy as np
import utils


def get_extractor_name(model_path):
    keywords = ["nonlinear", "linear", "spherical", "generative", "unet"]
    names = ["SNSE", "LSE", "SLSE", "CNSE", "UNet-512"]
    for i, k in enumerate(keywords):
        if k in model_path:
            return model_path#names[i]

def get_best_model(table, best_ind, name, delimiter=","):
    best_model = table.split("\n")[best_ind + 1]
    items = best_model.split(delimiter)
    return delimiter.join([name] + items)

def calc_subset(dic):
    indice = {}
    indice["face"] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]
    # eye glasses, ear ring, neck lace, hat, cloth
    indice["other"] = [3, 11, 12, 13, 15]
    for metric in ["AP", "AR", "IoU"]:
        for t in ["face", "other"]:
            arr = np.array(dic[metric])
            v = arr[indice[t]]
            v = v[v>-1]
            dic[f"m{metric}_{t}"] = v.mean()
    return dic


global_metrics = ["mAP", "mAR", "mIoU"]
for metric in ["AP", "AR", "IoU"]:
    for t in ["face", "other"]:
        global_metrics.append(f"m{metric}_{t}")


def process_dir(dir):
    files = glob.glob(f"{dir}/*agreement.npy")
    files.sort()
    print(files)

    global_table = []
    class_table = []
    global_tabular = []
    class_tabular = []
    for f in files:
        name = get_extractor_name(f)
        dic = np.load(f, allow_pickle=True)[()]
        dic["mIoU"][12] = -1
        dic = calc_subset(dic)
        res = utils.format_test_result(dic, global_metrics)
        global_csv, class_csv = res[2:]
        global_table.append(get_best_model(global_csv, 0, name))
        class_table.append(get_best_model(class_csv, 2, name))
        res = utils.format_test_result(dic, ["mIoU", "mIoU_face", "mIoU_other"])
        global_latex, class_latex = res[:2]
        global_tabular.append(get_best_model(global_latex, 1, name, " & "))
        class_tabular.append(get_best_model(class_latex, 3, name, " & "))
    global_csv_head = global_csv.split("\n")[0]
    class_csv_head = class_csv.split("\n")[0]
    global_latex_head = global_latex.split("\n")[:2]
    class_latex_head = class_latex.split("\n")[:2]
    l = [global_csv_head, global_table, class_csv_head, class_table]
    l.extend([global_latex_head, global_tabular, class_latex_head, class_tabular])
    return l

dir = "record/celebahq[0-1]"
dirs = glob.glob(dir)
dirs.sort()
global_table = []
class_table = []
global_latex = []
class_latex = []
for d in dirs:
    hg, tg, hc, tc, hlg, lg, hlc, lc = process_dir(d)
    global_table.extend(tg)
    class_table.extend(tc)
    global_latex.extend(lg)
    class_latex.extend(lc)
with open("celeba_global_result.csv", "w") as f:
    f.write("\n".join([hg] + global_table))
with open("celeba_class_result.csv", "w") as f:
    f.write("\n".join([hc] + class_table))
with open("celeba_agreement_tabular.tex", "w") as f:
    f.write("\n".join(hlc + class_latex + hlg + global_latex))
