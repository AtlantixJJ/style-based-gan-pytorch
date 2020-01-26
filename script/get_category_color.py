import sys
sys.path.insert(0, ".")
import utils

category = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

colors = utils.labelcolormap(len(category))
colors = ["rgb(%d, %d, %d)" % (c[0], c[1], c[2]) for c in colors]
print(category)
print(colors)
