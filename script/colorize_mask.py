import sys
sys.path.insert(0, ".")
import os
from PIL import Image
import glob
import numpy as np
import utils
#color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

basedir = sys.argv[1]
folder_base = basedir + '/CelebAMask-HQ-mask'
folder_save = basedir + '/CelebAMask-HQ-mask-viz'
img_num = len(os.listdir(folder_base))

os.system(f"mkdir {folder_save}")
colorizer = utils.Colorize(16)
for k in range(img_num):
    filename = os.path.join(folder_base, str(k) + '.png')		
    if os.path.exists(filename):
        im = np.array(Image.open(filename))
        im = utils.idmap(im)
        viz = colorizer(im)
    filename_save = os.path.join(folder_save, str(k) + '.png')
    result = Image.fromarray(viz.astype(np.uint8))
    print (filename_save)
    result.save(filename_save)

