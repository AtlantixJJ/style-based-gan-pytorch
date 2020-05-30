import sys
sys.path.insert(0, ".")
import utils

category = ['background', 'skin', 'nose', 'eye_g', 'eye', 'brow', 'ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck', 'cloth']
order = [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

colors = utils.labelcolormap(len(category))
js_colors = ["rgb(%d, %d, %d)" % (c[0], c[1], c[2]) for c in colors]
py_colors = ["(%d, %d, %d)" % (c[0], c[1], c[2]) for c in colors]
print(category)
print()
print(js_colors)
print()
print("[%s]" % ",".join(py_colors))

LABEL_COLORS = [
    [0, 0, 0,      ],
    [208, 2, 27,   ],
    [245, 166, 35, ],
    [248, 231, 28, ],
    [139, 87, 42,  ],
    [126, 211, 33, ],
    [255, 255, 255 ],
    [226, 238, 244,],
    [226, 178, 213,],
    [74, 144, 226, ],
    [80, 227, 194, ]]
TEMP_NAMES = [
    "background",
    "quilt",
    "pillow",
    "window",
    "curtain"
]