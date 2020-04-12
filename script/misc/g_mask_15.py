import sys
sys.path.insert(0, ".")
import os
import cv2
import glob
import numpy as np
import utils
import pprint 
#list2	 
# 5->4, 7->6, 9->8
label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
id2cid = utils.create_id2cid(19, map_from=[5, 7, 9, 16], map_to=[4, 6, 8, 17])
pprint.pprint(id2cid)
basedir = sys.argv[1]
folder_base = basedir + '/CelebAMask-HQ-mask'
folder_save = basedir + '/CelebAMask-HQ-mask-15'
img_num = 30000

os.system(f"mkdir {folder_save}")

for k in range(img_num):
	im_base = np.zeros((512, 512))
	filename = os.path.join(folder_base, str(k) + '.png')
	im = cv2.imread(filename)
	new_im = utils.idmap(im, id2cid)
	filename_save = os.path.join(folder_save, str(k) + '.png')
	cv2.imwrite(filename_save, new_im)

