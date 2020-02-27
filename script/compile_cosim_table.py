import numpy as np
import sys, glob, torch
sys.path.insert(0, ".")
import utils

import matplotlib.pyplot as plt
"""
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
"""

files = glob.glob("record/celebahq_cosim/*_[0-2]_cosim.npy")
files.sort()
cosim_table = np.load(files[0], allow_pickle=True)[()]
mean_table = cosim_table[0]
std_table = cosim_table[1]
cats = utils.CELEBA_REDUCED_CATEGORY
tables = [mean_table, std_table]
name = ["mean of cosine similarity", "standard deviation of cosine similarity", "log data number"]
fig = plt.figure(figsize=(15, 8))
for ind in range(len(tables)):
    ax = plt.subplot(1, len(tables), ind + 1)
    ax.imshow(tables[ind])
    ax.set_xticks(np.arange(len(cats)))
    ax.set_yticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yticklabels(cats)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    for i in range(len(cats)):
        for j in range(len(cats)):
            text = ax.text(j, i, "%.1f" % tables[ind][i, j],
                        ha="center", va="center", color="w")
    ax.set_title(name[ind])
fig.tight_layout()
plt.savefig("test.png", box_inches="tight")