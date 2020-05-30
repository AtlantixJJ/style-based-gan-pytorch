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


"""
@param:
    x : (N, C)
    w : (M, C)
"""
def logp_exp_gaussian(x, theta):
    p, lam, mu, sigma = theta
    prob_exp = p * lam * torch.exp(-lam * x)
    prob_gau = (1 - p) / math.sqrt(2 * math.pi) / sigma * torch.exp(- (x - mu) ** 2 / 2 / sigma / sigma)
    return torch.log(prob_exp + prob_gau)

"""
@param:
    x : (N, C)
    w : (M, C)
"""
def likelihood(x, theta, func):
    logs = func(x, theta) # (N, M)
    indice = torch.argmin(logs, 1, keepdim=True)
    mins = logs.gather(1, indice)
    return mins.sum()

def plot_scatter(x1, x2):
    plt.scatter(x1[:, 0], x1[:, 1], s=2, c="r")
    plt.scatter(x2[:, 0], x2[:, 1], s=2, c="b")

def plot_linear(x1, x2, w, b, xmin, xmax):
    plt.hist(x1)
    plt.hist(x2)
    #w = F.normalize(w, 2, 1)
    with torch.no_grad():
        coef = w[0] - w[1]
        bias = b[0] - b[1]
        x = torch.linspace(xmin, xmax, 100)
        y = -coef[0]/coef[1] * x - bias / coef[1]
        plt.plot(x, y)
    plt.axis([xmin, xmax])

N = 1000
M = 2
C = 2
torch.manual_seed(6)
x1 = torch.randn((N // 2,)) ** 2 
x2 = torch.randn((N,)) + 1
x = torch.cat([x1, x2])
xmin, xmax = x.min(), x.max()
p, lam, mu, sigma = torch.nn.Parameter(torch.Tensor([0.5, 1, 0.5, 3]))

theta = [p, lam, mu, sigma]
optim = torch.optim.Adam(theta, lr=1e-2)
best_likelihood = 0xffffff
best_theta = 0
improv = 100
count = 0

os.system("rm video/*.png")


while improv > 1e-3 and count < 500:
    optim.zero_grad()

    L = likelihood(x, theta, logp_exp_gaussian) / x.shape[0]

    L.backward()
    optim.step()

    improv = 0.9 * improv + 0.1 * max(best_likelihood - L, 0)
    if best_likelihood > L:
        best_likelihood = L
        best_theta = [t.detach().clone() for t in theta]

    if count % 10 == 0:
        plot_exp_gaussian(x1, x2, w.detach(), xmin, xmax)
        plt.title("Likelihood: %.2f" % L)
        plt.savefig(f"video/iter{count//10}.png")
        plt.close()

    print(f"=> Iteration {count}: Likelihood: {L:.3f}")
    count += 1

os.system("ffmpeg -y -f image2 -r 12 -i video/iter%d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")

w, b = best_theta
print(f"=> Best likelihood: {best_likelihood:.1f}")