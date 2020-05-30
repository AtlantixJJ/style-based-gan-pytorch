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
def logp_linear(x, theta, eps=1e-6):
    w, b = theta
    #w = F.normalize(w, 2, 1)
    logit = torch.matmul(x, w.permute(1, 0)) + b.permute(1, 0) # (N, M)
    # take log-probability
    p = torch.exp(logit)
    p = p / (p.sum(1, keepdim=True) + eps)
    return -torch.log(p) # (N, M)

def logp_euc(x, theta, eps=1e-6):
    w, b = theta
    d = torch.stack([((x - w[i:i+1]) ** 2).sum(1)
        for i in range(w.shape[0])], 1)
    return d

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

def plot_voroni(x1, x2, w, xmin, xmax, ymin, ymax):
    plot_scatter(x1, x2)

    direction = w[1] - w[0]
    mid = direction / 2 + w[0]
    k = - direction[0] / direction[1]
    x = torch.linspace(xmin, xmax, 100)
    y = k * (x - mid[0]) + mid[1]
    plt.plot(x, y)
    plt.scatter(w[0, 0], w[0, 1], s=10)
    plt.scatter(w[1, 0], w[1, 1], s=10)
    plt.axis([xmin, xmax, ymin, ymax])

def plot_linear(x1, x2, w, b, xmin, xmax, ymin, ymax):
    x = torch.cat([x1, x2])
    l = logp_linear(x, [w, b]).argmax(1)
    c = [colors[i + 10] for i in l]
    plt.scatter(x[:, 0], x[:, 1], c=c)
    #w = F.normalize(w, 2, 1)
    with torch.no_grad():
        coef = w[0] - w[1]
        bias = b[0] - b[1]
        x = torch.linspace(xmin, xmax, 100)
        y = -coef[0]/coef[1] * x - bias / coef[1]
        plt.plot(x, y)
    plt.axis([xmin, xmax, ymin, ymax])

N = 1000
M = 2
C = 2
torch.manual_seed(6)
x1 = torch.randn(N, 2)
x1 = torch.matmul(x1, torch.randn(2, 2)) + torch.randn(1, 2) * 2
x2 = torch.randn(N, 2)
x2 = torch.matmul(x2, torch.randn(2, 2)) + torch.randn(1, 2) * 2
x = torch.cat([x1, x2])
xmin, xmax = x[:, 0].min(), x[:, 0].max()
ymin, ymax = x[:, 1].min(), x[:, 1].max()
w = torch.nn.Parameter(torch.randn(M, C))
b = torch.nn.Parameter(torch.randn(M, 1) * 0.1)
theta = [w, b]
optim = torch.optim.Adam([w], lr=1e-2)
best_likelihood = 0xffffff
best_theta = 0
improv = 100
count = 0

os.system("rm video/*.png")

t = "linear"

def rot(t):
    return torch.Tensor(
        [
            [math.cos(t), math.sin(t)],
            [-math.sin(t), math.cos(t)]
        ])


while improv > 1e-3 and count < 500:
    #optim.zero_grad()

    L = 0
    regloss = 0
    if t == "euc":
        L = L + likelihood(x, theta, logp_euc) / x.shape[0]
        diff_w = w[1:, :] - w[:-1, :]
        diff_w_l2 = (diff_w ** 2).sum(1)
        regloss = regloss + 100 * ((diff_w_l2 - 1) ** 2).sum()
    elif t == "linear":
        L = L + likelihood(x, theta, logp_linear) / x.shape[0]

    #(L + regloss).backward()
    #optim.step()

    w = w.matmul(rot(0.01))
    w.grad = torch.zeros_like(w)
    theta = [w, b]

    improv = 0.9 * improv + 0.1 * max(best_likelihood - L, 0)
    if best_likelihood > L:
        best_likelihood = L
        best_theta = [w.detach(), b.detach()]

    if count % 10 == 0:
        if t == "euc":
            plot_voroni(x1, x2, w.detach(), xmin, xmax, ymin, ymax)
        elif t == "linear":
            plot_linear(x1, x2, w.detach(), b.detach(), xmin, xmax, ymin, ymax)
        plt.title("Likelihood: %.2f" % L)
        plt.savefig(f"video/iter{count//10}.png")
        plt.close()

    print(f"=> Iteration {count}: Likelihood: {L:.3f} Reg: {regloss:.3f} Grad: [{w.grad.min():.3f}, {w.grad.max():.3f}]")
    count += 1

os.system("ffmpeg -y -f image2 -r 12 -i video/iter%d.png -pix_fmt yuv420p -b:v 16000k demo.mp4")

w, b = best_theta
print(f"=> Best likelihood: {best_likelihood:.1f}")