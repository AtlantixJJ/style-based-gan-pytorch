import numpy as np
import matplotlib.pyplot as plt
import sys

WINDOW_SIZE = 40
name = sys.argv[1] # fixseg_1.0_mul-16
trace_path = f"expr/{name}/trace_weight.npy"

trace = np.load(trace_path) # (N, 16, D)
weight = trace[-1]
weight_min, weight_max = weight.min(), weight.max()
moving_avg_trace = np.zeros_like(trace)
step_delta = np.zeros_like(trace)

# weight vector
fig = plt.figure(figsize=(12, 12))
maximum, minimum = weight.max(), weight.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(weight[j])
    ax.axes.get_xaxis().set_visible(False)
    #ax.set_ylim([weight[j].min(), weight[j].max()])
    ax.set_ylim([weight_min, weight_max])
fig.savefig(f"results/{name}_weight.png", bbox_inches='tight')
plt.close()

for i in range(trace.shape[0]):
    bg, ed = max(0, i - WINDOW_SIZE // 2), min(i + WINDOW_SIZE // 2, trace.shape[0])
    moving_avg_trace[i] = trace[bg:ed].mean(0)

step_delta[1:] = trace[1:] - trace[:-1]
step_delta[0] = 0
step_delta_norm = np.linalg.norm(step_delta, ord=2, axis=1)

fig = plt.figure(figsize=(10, 10))
maximum, minimum = step_delta_norm.max(), step_delta_norm.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(step_delta_norm[:, j])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_step_delta_norm.png", bbox_inches='tight')
plt.close()

moving_avg_delta = trace - moving_avg_trace
moving_avg_delta_norm = np.linalg.norm(moving_avg_delta, ord=2, axis=1)

fig = plt.figure(figsize=(10, 10))
maximum, minimum = moving_avg_delta_norm.max(), moving_avg_delta_norm.min()
for j in range(16):
    ax = plt.subplot(4, 4, j + 1)
    ax.plot(moving_avg_delta_norm[:, j])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([minimum, maximum])
fig.savefig(f"results/{name}_moving_avg_delta_norm.png", bbox_inches='tight')
plt.close()


