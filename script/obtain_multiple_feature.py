import os

for i in range(10):
    os.system("python script/obtain_rcc_feature.py --name kmeans_feats_%d" % i)