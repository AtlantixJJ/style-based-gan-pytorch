import sys
sys.path.insert(0, ".")
import utils
import numpy as np
import glob


def get_name(fname):
    ind1 = fname.rfind("/")
    return fname[ind1+1:]


def get_best_model(table, best_ind, name):
    best_model = table.split("\n")[best_ind + 1]
    items = best_model.split(",")
    items[0] = name
    return ",".join(items)


dir = sys.argv[1]
files = glob.glob(f"{dir}/*agreement.npy")
files.sort()
print(files)

names = []
dic = {}
for f in files:
    name = get_name(f)
    names.append(name)
    words = name.split("_")
    for word in words:
        try:
            dic[word] += 1
        except:
            dic[word] = 1

for k in list(dic.keys()):
    if dic[k] != len(files):
        del dic[k]

for i in range(len(names)):
    words = names[i].split("_")
    words = [w for w in words if w not in dic.keys()]
    names[i] = "_".join(words)

best_models = []
best_class = []
for i in range(len(files)):
    dic = np.load(files[i], allow_pickle=True)[()]
    name = names[i]
    _, _, model_global_csv, class_csv = utils.format_test_result(dic)
    best_models.append(get_best_model(model_global_csv, 0, name))
    best_class.append(get_best_model(class_csv, 2, name))
best_models = [model_global_csv.split("\n")[0]] + best_models
best_class = [class_csv.split("\n")[0]] + best_class

name = dir.replace("record/", "")

with open(f"{name}_global.csv", "w") as f:
    f.write("\n".join(best_models))
with open(f"{name}_class.csv", "w") as f:
    f.write("\n".join(best_class))

