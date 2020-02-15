import numpy as np
import matplotlib.pyplot as plt
import sys

segments = np.cumsum([512, 512, 512, 512, 256, 128, 64, 32, 16])

WINDOW_SIZE = 100
file_path = sys.argv[1] # fixseg_1.0_mul-16
ind = file_path.rfind("/")
name = file_path[ind + 1:].replace("fixseg_1.0_", "").replace("_contrib.npy", "")

# data
data = np.load(file_path) # (N, 16, D)

# variance
var = data.std(0)
fig = plt.figure(figsize=(12, 12))
maximum, minimum = var.max(), var.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(var[j])
    for x in segments:
        ax.axvline(x=x)
    ax.axes.get_xaxis().set_visible(False)
    #ax.set_ylim([weight[j].min(), weight[j].max()])
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_contrib_var.png", bbox_inches='tight')
plt.close()


# weight vector
fig = plt.figure(figsize=(12, 12))
mean = data.mean(0)
maximum, minimum = mean.max(), mean.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(mean[j])
    for x in segments:
        ax.axvline(x=x)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_contrib_mean.png", bbox_inches='tight')
plt.close()