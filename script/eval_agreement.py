"""
Format the result of agreement evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import sys

npyfile = sys.argv[1]

label_list = ['skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
label_latex = [l.replace("_", "\\_") for l in label_list]

global_metrics = ["pixelacc", "mAP", "mAR", "mIoU"]
class_metrics = ["AP", "AR", "IoU"]

def str_num(n):
    return ("%.3f" % n).replace(".000", "")

def str_latex_table(strs):
    ncols = len(strs[0])
    seps = "".join(["c" for i in range(ncols)])
    s = []
    s.append("\\begin{tabular}{%s}" % seps)
    s.append(" & ".join(strs[0]) + " \\\\\\hline")
    for line in strs[1:]:
        s.append(" & ".join(line) + " \\\\")
    s.append("\\end{tabular}")
    return "\n".join(s)

dic = np.load(npyfile, allow_pickle=True)[()]
n_model = len(dic["mIoU"])
iters = [i * 1000 for i in range(1, 1 + n_model)]

# table 1: model iterations and global accuracy
numbers = [iters, dic["pixelacc"], dic["mAP"], dic["mAR"], dic["mIoU"]]
numbers = np.array(numbers).transpose() # (10, 5)
strs = [["step"] + global_metrics]
for i in range(n_model):
    strs.append([str_num(n[i]) for n in numbers[i]])
    strs[-1][0] = strs[-1][0][:-4]
numbers = np.array(numbers)
# print latex table
print(str_latex_table(strs))

# table 2: classwise accuracy
best_ind = np.argmax(dic["mIoU"])
strs = [["metric"] + label_latex]
numbers = []
for metric in class_metrics:
    data = dic[metric][best_ind][1:] # ignore background
    numbers.append(data)
numbers = np.array(numbers) # (3, 16)
for i in range(len(class_metrics)):
    strs.append(["%.3f" % n if n > 0 else "-" for n in numbers[i]])
for i in range(1, len(strs)):
    strs[i] = [class_metrics[i - 1]] + strs[i]
# print latex table
print(str_latex_table(strs))