import sys
sys.path.insert(0, '..')
import numpy as np
import torch

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        if gray_image.shape[0] == 1:
            gray_image = gray_image[0]
        size = gray_image.shape
        assert len(size) == 2
        h, w = size

        if isinstance(gray_image, torch.Tensor):
            color_image = torch.zeros(3, h, w, device=gray_image.device).fill_(0)
            for label in range(0, len(self.cmap)):
                mask = (label == gray_image).cpu()
                color_image[0][mask] = int(self.cmap[label, 0])
                color_image[1][mask] = int(self.cmap[label, 1])
                color_image[2][mask] = int(self.cmap[label, 2])
        else:
            color_image = np.zeros((h, w, 3), dtype="uint8")
            for label in range(len(self.cmap)):
                mask = (label == gray_image)
                color_image[mask] = self.cmap[label]
        return color_image


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    """
    edge_num = int(np.ceil(np.power(N + , 1/3))) - 1
    cmap = np.zeros((N, 3), dtype=np.uint8)
    step_size = 255. / edge_num
    cmap[0] = (0, 0, 0)
    count = 1
    for i in range(edge_num + 1):
        for j in range(edge_num + 1):
            for k in range(edge_num + 1):
                if count >= N or (i == j and j == k):
                    continue
                cmap[count] = [int(step_size * n) for n in [i, j, k]]
                count += 1
    """

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap