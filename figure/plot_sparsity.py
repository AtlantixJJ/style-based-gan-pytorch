import sys
sys.path.insert(0, ".")
import numpy as np
import glob
import utils
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

def process(list):
    list = np.array(list)
    return list / list.max()

table = open("record/sparsity.csv", "r")
sparsity = []
miou = []
for line in table.readlines():
    items = [float(i.strip()) for i in line.split(",")]
    sparsity.append(items[0])
    miou.append(items[1])
data = list(zip(sparsity, miou))
data.sort(key=lambda x : x[0])
data = np.array(data)
sparsity = data[:, 0] / data[:, 0].max()
miou = data[:, 1] / data[:, 1].max()
line = np.polyfit(sparsity[:5], miou[:5])
plt.plot(np.arange(5), )
plt.scatter(sparsity, miou)
plt.savefig('test.png', box_inches="tight")
plt.close()