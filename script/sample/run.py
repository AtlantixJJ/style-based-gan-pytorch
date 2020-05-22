import os

def command_snapshot(gpus):
    count = 0
    basecmd = "python script/sample/plot.py --seed %d"
    for s in [65537, 1314, 1973, 3771]:
        idx = count % len(gpus)
        yield idx, basecmd % s
        count += 1

def command_analyze(gpus):
    count = 0
    basecmd = "python script/sample/visualize_latent_trace.py --seed %d"
    for s in [65537, 1314, 1973, 3771]:
        idx = count % len(gpus)
        yield idx, basecmd % s
        count += 1


def command_fewshot(gpus):
    count = 0
    basecmd = "python script/sample/ms.py --kl-coef 0 --n-iter 1000 --seed %d --gpu %d --outdir results/mask_fewshot_%d --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model"
    for t in [1, 2, 4, 8]:
        for s in [65537, 1314, 1973, 3771]:
            idx = count % len(gpus)
            yield idx, basecmd % (s, gpus[idx], t, t)
            count += 1
gpus = [0]

def command_sample(gpus):
    count = 0
    basecmd = "python script/sample/ms.py --kl-coef 0 --n-iter 1000 --seed %d --gpu %d"
    for s in [65537, 1314, 1973, 3771]:
        idx = count % len(gpus)
        yield idx, basecmd % (s, gpus[idx])
        count += 1
gpus = [0, 1, 2, 3]


slots = [[] for _ in gpus]
for i, c in command_sample(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
