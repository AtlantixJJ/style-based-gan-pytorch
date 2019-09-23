import torch


def select_max(x, y):
    if x < y:
        return y
    else:
        return x
