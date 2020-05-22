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
