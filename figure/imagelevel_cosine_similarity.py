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

files = glob.glob("record/celebahq_cosim/*_cosim.npy")
files.sort()
mean_tables = []
std_tables = []
size_tables = []
for i, f in enumerate(files):
    if i > 25:
        break
    cosim_table = np.load(f, allow_pickle=True)[()]
    mean_table, std_table, size_table = cosim_table
    size_table /= 10000.0
    mean_tables.append(mean_table)
    std_tables.append(std_table)
    size_tables.append(size_table)

mean_tables = np.array(mean_tables)
std_tables = np.array(std_tables)
size_tables = np.array(size_tables)
size_tables = size_tables / size_tables.sum(0, keepdims=True)
size_tables = np.nan_to_num(size_tables)
mean_table = (mean_tables * size_tables).sum(0)
std_table = (std_tables * size_tables).sum(0)

cats = utils.CELEBA_REDUCED_CATEGORY
# remove necklace because it is 0 everywhere
del cats[13]

def remove_numpy(x, idx):
    x = np.concatenate([x[:idx], x[idx+1:]], 0)
    x = np.concatenate([x[:, :idx], x[:, idx+1:]], 1)
    return x

mean_table = remove_numpy(mean_table, 13)
std_table = remove_numpy(std_table, 13)
tables = [mean_table, std_table]
name = ["mean of cosine similarity", "standard deviation of cosine similarity"]
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
plt.savefig("imagelevel_cosine_similarity.png", box_inches="tight")