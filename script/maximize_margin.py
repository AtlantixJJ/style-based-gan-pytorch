import torch
import torch.nn.functional as F
import numpy as np
import os, math
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
import matplotlib.pyplot as plt


def solve_gaussian_intersection(m1, m2, s1, s2):
    if s1 == s2:
        return (m1 + m2) / 2
    switch = False
    if s1 < s2:
        switch = True
        s1, s2 = s2, s1
        m1, m2 = m2, m1

    s12 = s1 ** 2
    s22 = s2 ** 2
    const = s12 * m2 - s22 * m1
    log = torch.log(s1 / s2)
    sqrt = 2 * (s12 - s22) * log + (m1 - m2) ** 2
    sqrt = torch.sqrt(sqrt)
    sqrt = sqrt * s1 * s2
    div = s12 - s22

    plus = True
    if m1 < m2:
        plus = False
    
    if switch:
        s1, s2 = s2, s1
        m1, m2 = m2, m1 

    if plus:
        return (const + sqrt) / div
    else:
        return (const - sqrt) / div


def phi(x, m, s):
    input = torch.Tensor([(x - m) / (s * math.sqrt(2))])
    erf = torch.erf(input)
    return 0.5 + 0.5 * erf


def log_double_phi(x, m1, m2, s1, s2):
    l = -torch.log(phi(2 * m1 - x, m1, s1) + phi(x, m2, s2))
    return l