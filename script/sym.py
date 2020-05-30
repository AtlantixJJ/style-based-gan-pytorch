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
m2 = sympy.Symbol('m_2') # need m1 > m2
sigma1 = sympy.Symbol('\sigma_1')
sigma2 = sympy.Symbol('\sigma_2')
eq = sympy.Eq(sympy.ln(sigma1/sigma2) + (x - m1) ** 2 / (2 * sigma1 ** 2) - (x - m2) ** 2 / (2 * sigma2 ** 2), 0)
margin_x = sympy.solve(eq, x)[0]
phi = lambda i : 0.5 + 0.5 * sympy.erf(i / math.sqrt(2))
margin = -sympy.ln(
    phi((2 * m1 - margin_x) / sigma1) + 
    phi((margin_x - m2) / sigma2))
mx = sympy.Symbol('mx')
margin_subs = -sympy.ln(
    phi((2 * m1 - mx) / sigma1) + 
    phi((mx - m2) / sigma2))
p1 = 1 / (sympy.sqrt(2 * sympy.pi) * sigma1) * sympy.exp(-1/(2 * sigma1 ** 2) * (x - m1) ** 2)
p2 = 1 / (sympy.sqrt(2 * sympy.pi) * sigma2) * sympy.exp(-1/(2 * sigma2 ** 2) * (x - m2) ** 2)


def plot_sol(m1_, sigma1_, m2_, sigma2_, mini, maxi, sol=None):
    xs = np.linspace(mini, maxi, 100)
    y1 = [p1.evalf(subs={x:x_,m1:m1_,sigma1:sigma1_}) for x_ in xs]
    y2 = [p2.evalf(subs={x:x_,m2:m2_,sigma2:sigma2_}) for x_ in xs]
    full_dict = {m1:m1_,sigma1:sigma1_,m2:m2_,sigma2:sigma2_}
    M = 0
    if sol is None:
        sol = margin_x.evalf(subs=full_dict)
        M = margin.evalf(subs=full_dict)
    else:
        full_dict.update({mx:sol})
        M = margin_subs.evalf(subs=full_dict)
    plt.axvline(x=sol, ls="--")
    plt.plot(xs, y1)
    plt.plot(xs, y2)
    plt.axis([mini, maxi, 0, 1])
    plt.title("Margin %.3f" % M)

m1_, m2_ = (np.random.randn(2) * 2).tolist()
sigma1_, sigma2_ = (np.random.randn(2) ** 2 * 2).tolist()

m1_ = 0; sigma1_ = 1
m2s =     [1, 1, 1, 1, 1, -1, 1, 2]
sigma2s = [1.1, 0.2, 1.1, 5, 1.1, 0.6, 0.6, 0.6]

step = 1/24./2

def interpolate(x1, x2, y1, y2):
    ts = np.arange(0, 1, step)
    return [t * (y2 - y1) / (x2 - x1) + y1
        for t in ts]

os.system("rm video/*.png")
count = 0

mini, maxi = -3, 3

# output sol scan
for sol in np.arange(-2, 2, step):
    plot_sol(0, 1, 1, 1.1, mini, maxi, sol=sol)
    plt.savefig(f"video/{count}.png")
    plt.close()
    count += 1

for i in range(len(m2s) - 1):
    ms = interpolate(0, 1, m2s[i], m2s[i+1])
    ss = interpolate(0, 1, sigma2s[i], sigma2s[i+1])
    for m2_, sigma2_ in zip(ms, ss):
        if m1_ > m2_:
            plot_sol(m2_, sigma2_, m1_, sigma1_, mini, maxi)
        else:
            plot_sol(m1_, sigma1_, m2_, sigma2_, mini, maxi)
        plt.savefig(f"video/{count}.png")
        plt.close()
        count += 1
os.system("ffmpeg -y -f image2 -r 24 -i video/%d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")