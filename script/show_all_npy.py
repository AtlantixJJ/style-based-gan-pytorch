import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils


files = glob.glob("results/var_metric/*.npy")

for f in files:
    arr = np.load(f, allow_pickle=True)[()]
    utils.plot_dic(arr, f, f.replace(".npy", ".png"))
