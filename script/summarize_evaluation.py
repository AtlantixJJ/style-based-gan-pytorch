import sys
sys.path.insert(0, ".")
import utils
import numpy as np
import glob


def get_name(fname):
    fname_ = fname.replace("_agreement.npy", "")
    fname_ = fname_.replace("celebahq_stylegan_linear_layer0,1,2,3,4,5,6,7,8_vbs8_", "")
    ind1 = fname_.rfind("/")
    return fname_[ind1+1:]


def get_best_model(table, best_ind, name):
    best_model = table.split("\n")[best_ind + 1]
    items = best_model.split(",")
    items[0] = name
    return ",".join(items)


dir = sys.argv[1]
files = glob.glob(f"{dir}/*agreement.npy")
files.sort()
print(files)

best_models = []
best_class = []
for f in files:
    dic = np.load(f, allow_pickle=True)[()]
    name = get_name(f)
    _, _, model_global_csv, class_csv = utils.format_test_result(dic)
    best_models.append(get_best_model(model_global_csv, 0, name))
    best_class.append(get_best_model(class_csv, 2, name))
best_models = [model_global_csv.split("\n")[0]] + best_models
best_class = [class_csv.split("\n")[0]] + best_class

with open("global_result.csv", "w") as f:
    f.write("\n".join(best_models))
with open("class_result.csv", "w") as f:
    f.write("\n".join(best_class))

