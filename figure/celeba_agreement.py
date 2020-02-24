import sys, glob
sys.path.insert(0, ".")
import numpy as np
import utils


dir = sys.argv[1]
files = glob.glob(f"{dir}/*agreement.npy")
files.sort()
print(files)
dic = np.load(files[0], allow_pickle=True)[()]

def get_name(fname):
    fname_ = fname.replace("_agreement.npy", "")
    fname_ = fname_.replace("-16", "")
    ind1 = fname_.rfind("/")
    return fname_[ind1+1:]

def get_best_model(table, best_ind, name):
    best_model = table.split("\n")[best_ind + 1]
    items = best_model.split(",")
    items[0] = name
    return ",".join(items)

def calc_subset(dic):
    indice = {}
    indice["face"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
    indice["other"] = [11, 12, 13, 15]
    for metric in ["AP", "AR", "IoU"]:
        for t in ["face", "other"]:
            arr = np.array(dic[metric])
            v = arr[indice[t]]
            v = v[v>-1]
            dic[f"m{metric}_{t}"] = v.mean()
    return dic

global_metrics = ["mAP", "mAR", "mIoU"]
best_models = []
best_class = []
for metric in ["AP", "AR", "IoU"]:
    for t in ["face", "other"]:
        global_metrics.append(f"m{metric}_{t}")

for f in files:
    name = get_name(f)
    dic = np.load(f, allow_pickle=True)[()]
    dic = calc_subset(dic)
    res = utils.format_test_result(dic, global_metrics)
    global_latex, class_latex, global_csv, class_csv = res
    best_models.append(get_best_model(global_csv, 0, name))
    best_class.append(get_best_model(class_csv, 2, name))
with open("global_result.csv", "w") as f:
    f.write("\n".join(best_models))
with open("class_result.csv", "w") as f:
    f.write("\n".join(best_class))