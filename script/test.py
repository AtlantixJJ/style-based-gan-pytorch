import torch, math
EPS = 1e-8

def sphere2cartesian(x): #x: (N, C)
    r = x[:, 0:1]
    si = torch.sin(x)
    si[:, 0] = 1
    si = torch.cumprod(si, 1)
    co = torch.cos(x)
    co[:, 0] = 1
    co = torch.roll(co, -1, 1)
    return si * co * r

def arccot(x):
    # torch.atan = np.arctan
    return math.pi / 2 - torch.atan(x)

def cartesian2sphere(x): #x: (N, C)
    # (xn, xn + xn-1, ..., xn + xn-1 + ... + x2 + x1)
    cx = EPS + torch.sqrt(torch.cumsum(torch.flip(x, [1]) ** 2, 1))
    # (xn + xn-1 +...+ x2 + x1, xn + xn-1 +...+ x2, ..., xn + xn-1, xn)
    cx = torch.flip(cx, [1])
    r = cx[:, 0:1] #(N, 1)
    phi_1_n2 = arccot(x[:, :-2] / cx[:, 1:-1]) # (N, C-2)
    phi_n1 = 2 * arccot((x[:, -2] + cx[:, -2]) / x[:, -1])
    phi_n1 = phi_n1.view(-1, 1) # (N, 1)
    return torch.cat([r, phi_1_n2, phi_n1], 1)


"""
def sphere2cartesian(x): #x: (N, C)
    r = x[:, 0:1]
    si = np.sin(x)
    si[:, 0] = 1
    si = np.cumprod(si, 1)
    co = np.cos(x)
    co[:, 0] = 1
    co = np.roll(co, -1, 1)
    return si * co * r

def arccot(x):
    return np.pi / 2 - np.arctan(x)

def cartesian2sphere(x): #x: (N, C)
    # (xn, xn + xn-1, ..., xn + xn-1 + ... + x2 + x1)
    cx = EPS + np.sqrt(np.cumsum(x[:, ::-1] ** 2, 1))
    # (xn + xn-1 +...+ x2 + x1, xn + xn-1 +...+ x2, ..., xn + xn-1, xn)
    cx = cx[:, ::-1]
    r = cx[:, 0:1] #(N, 1)
    phi_1_n2 = arccot(x[:, :-2] / cx[:, 1:-1]) # (N, C-2)
    phi_n1 = 2 * arccot((x[:, -2] + cx[:, -2]) / x[:, -1])
    phi_n1 = np.expand_dims(phi_n1, 1) # (N, 1)
    return np.concatenate([r, phi_1_n2, phi_n1], 1)
"""

#a = np.random.randn(10000, 2544)
a = torch.randn(10000, 2544).cuda()
sphere_a = cartesian2sphere(a)
rec_a = sphere2cartesian(sphere_a)
rec_sphere_a = cartesian2sphere(rec_a)
#print(np.abs(a - rec_a).mean())
#print(np.abs(sphere_a - rec_sphere_a).mean())
diff = (a - rec_a).abs()
print(diff.mean(), diff.max())
diff = (sphere_a - rec_sphere_a).abs()
print(diff.mean(), diff.max())