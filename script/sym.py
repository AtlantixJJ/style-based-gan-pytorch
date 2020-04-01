import sympy
import torch
import math, os
import numpy as np
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())
import matplotlib.pyplot as plt


x = sympy.Symbol('x')
m1 = sympy.Symbol('m_1')
m2 = sympy.Symbol('m_2')
sigma1 = sympy.Symbol('\sigma_1')
sigma2 = sympy.Symbol('\sigma_2')
eq = sympy.Eq(sympy.ln(sigma1/sigma2) + (x - m1) ** 2 / (2 * sigma1 ** 2) - (x - m2) ** 2 / (2 * sigma2 ** 2), 0)
#res = sympy.solve(eq, x)
#print(sympy.latex(res))
p1 = 1 / (sympy.sqrt(2 * sympy.pi) * sigma1) * sympy.exp(-1/(2 * sigma1 ** 2) * (x - m1) ** 2)
p2 = 1 / (sympy.sqrt(2 * sympy.pi) * sigma2) * sympy.exp(-1/(2 * sigma2 ** 2) * (x - m2) ** 2)


def solve_gaussian_intersection(m1, m2, s1, s2):
    if s1 == s2:
        return (m1 + m2) / 2

    if m1 < m2:
        m1, m2 = m2, m1
        s1, s2 = s2, s1

    s12 = s1 ** 2
    s22 = s2 ** 2
    const = s12 * m2 - s22 * m1
    log = torch.log(s1 / s2)
    sqrt = 2 * (s12 - s22) * log + (m1 - m2) ** 2
    sqrt = torch.sqrt(sqrt)
    sqrt = sqrt * s1 * s2
    div = s12 - s22

    return (const + sqrt) / div


def phi(x, m, s):
    input = (x - m) / (s * math.sqrt(2))
    erf = torch.erf(input)
    return 0.5 + 0.5 * erf


def log_double_phi(x, m1, m2, s1, s2):
    if m1 < m2:
        return -torch.log(phi(2 * m1 - x, m1, s1) + phi(x, m2, s2))
    else:
        return -torch.log(phi(x, m1, s1) + phi(2 * m2 - x, m2, s2))


def plot_sol(m1_, sigma1_, m2_, sigma2_, sol=None):
    mini = m1_ - sigma1_ * 4 #min(m1_ - sigma1_ * 3, m2_ - sigma2_ * 3)
    maxi = m1_ + sigma1_ * 4 #max(m2_ + sigma2_ * 3, m2_ + sigma2_ * 3)
    xs = np.linspace(mini, maxi, 100)
    y1 = [p1.evalf(subs={x:x_,m1:m1_,sigma1:sigma1_}) for x_ in xs]
    y2 = [p2.evalf(subs={x:x_,m2:m2_,sigma2:sigma2_}) for x_ in xs]
    full_dict = {m1:m1_,sigma1:sigma1_,m2:m2_,sigma2:sigma2_}
    m1_, sigma1_, m2_, sigma2_ = [torch.Tensor([x]) for x in [m1_, sigma1_, m2_, sigma2_]]
    if sol is None:
        sol = float(solve_gaussian_intersection(
            m1_, m2_, sigma1_, sigma2_))
    phi = log_double_phi(sol, m1_, m2_, sigma1_, sigma2_)
    plt.axvline(x=sol, ls="--")
    plt.plot(xs, y1)
    plt.plot(xs, y2)
    plt.axis([mini, maxi, 0, 1])
    plt.title("Margin %.3f" % phi)

#m1_, m2_ = (np.random.randn(2) * 2).tolist()
#sigma1_, sigma2_ = (np.random.randn(2) ** 2 * 2).tolist()

m1_ = 0; sigma1_ = 1
m2s =     [1, 1, 1, 1, 1, -1, 1, 2]
sigma2s = [1, 0.2, 1, 5, 1, 0.6, 0.6, 0.6]

step = 1/24./2

def interpolate(x1, x2, y1, y2):
    ts = np.arange(0, 1, step)
    return [t * (y2 - y1) / (x2 - x1) + y1
        for t in ts]

os.system("rm video/*.png")
count = 0

# output sol scan
for sol in np.arange(-2, 2, step):
    plot_sol(0, 1, 1, 1, sol=sol)
    plt.savefig(f"video/{count}.png")
    plt.close()
    count += 1

for i in range(len(m2s) - 1):
    ms = interpolate(0, 1, m2s[i], m2s[i+1])
    ss = interpolate(0, 1, sigma2s[i], sigma2s[i+1])
    for m2_, sigma2_ in zip(ms, ss):
        plot_sol(m1_, sigma1_, m2_, sigma2_)
        plt.savefig(f"video/{count}.png")
        plt.close()
        count += 1
os.system("ffmpeg -y -f image2 -r 24 -i video/%d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")