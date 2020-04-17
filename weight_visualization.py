import torch
import utils
import matplotlib.pyplot as plt

def find_min_max_weight(module):
    vals = []
    for i, conv in enumerate(module):
        w = utils.torch2numpy(conv[0].weight)
        vals.append(w.min())
        vals.append(w.max())
    return min(vals), max(vals)

def concat_weight(module):
    vals = []
    ws = []
    for i, conv in enumerate(module):
        w = conv[0].weight[:, :, 0, 0]
        ws.append(w)
    ws = torch.cat(ws, 1)
    return ws

def plot_weight_layerwise(module, minimum=-1, maximum=1, savepath="", subfix=""):
    for i, conv in enumerate(module):
        w = utils.torch2numpy(conv[0].weight)[:, :, 0, 0]

        fig = plt.figure(figsize=(16, 12))
        for j in range(16):
            ax = plt.subplot(4, 4, j + 1)
            ax.scatter(list(range(len(w[j]))), w[j], marker='.', s=20)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_ylim([minimum, maximum])
        plt.tight_layout()
        fig.savefig(f"{savepath}_l{i}{subfix}.png", bbox_inches='tight')
        plt.close()

def plot_weight_concat(w, minimum=-1, maximum=1, savepath="", subfix=""):
    fig = plt.figure(figsize=(16, 12))
    for j in range(w.shape[0]):
        ax = plt.subplot(4, 4, j + 1)
        ax.scatter(list(range(len(w[j]))), w[j], marker='.', s=20)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylim([minimum, maximum])
    plt.tight_layout()
    fig.savefig(f"{savepath}_weight_{subfix}.png", bbox_inches='tight')
    plt.close()

def get_norm_layerwise(module, minimum=-1, maximum=1, subfix=""):
    norms = []
    for i, conv in enumerate(module):
        w = conv[0].weight[:, :, 0, 0]
        norms.append(w.norm(2, dim=1))
    return utils.torch2numpy(torch.stack(norms))

def get_projection(data, vx, vy):
    return data.matmul(vx), data.matmul(vy)

def weight_project_direction(ws):
    xs = []
    ys = []
    for w in ws:
        wn = w / w.norm(2)
        x = torch.randn(w.shape[0])
        x -= wn * wn.dot(x) # ortho to w
        x /= x.norm(2)
        y = torch.randn(w.shape[0])
        y -= wn * wn.dot(y) # ortho to w
        y -= x * x.dot(y) # ortho to x
        y /= y.norm(2)
        xs.append(x)
        ys.append(y)
    return xs, ys

def projection_direction(size):
    x = torch.randn(size)
    x /= x.norm(2)
    y = torch.randn(size)
    y -= x * x.dot(y)
    y /= y.norm(2)
    return x, y

def get_random_projection(data):
    vx, vy = projection_direction(data.shape[1])

    return get_projection(data, vx, vy)

def weight_surgery(state_dict, func):
    for k,v in state_dict.items():
        state_dict[k] = func(v)

def early_layer_surgery(state_dict, st=[0]):
    for i in st:
        k = list(state_dict.keys())[i]
        state_dict[k] = state_dict[k].fill_(0)

def small_negative(x, margin=0.1):
    x[(x<0)&(x>-margin)]=0
    return x

def negative(x):
    x[x<0]=0
    return x

def small_absolute(x, margin=0.05):
    x[(x<margin)&(x>-margin)]=0
    return x

