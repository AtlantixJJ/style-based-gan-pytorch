import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import sys, os, glob
sys.path.insert(0, ".")

dics = glob.glob("eval_trace*.npy")
dics.sort()
dics = list(np.concatenate([np.load(d, allow_pickle=True) for d in dics]))

mIoU = [d["mIoU"] for d in dics]
mIoU_face = [d["mIoU_face"] for d in dics]
mIoU_other = [d["mIoU_other"] for d in dics]

fig = plt.figure(figsize=(10, 4))
plt.plot(mIoU)
plt.plot(mIoU_face)
plt.plot(mIoU_other)
plt.axis([-500, 11000, 0, 0.85])
plt.legend(["all", "face", "other"])
plt.savefig("eval_trace.pdf")
plt.close()

