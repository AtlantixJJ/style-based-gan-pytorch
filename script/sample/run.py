import os

def command_sample(gpus):
    count = 0
    basecmd = "python script/sample/ms.py --kl-coef 0 --n-iter 1600 --seed %d --gpu %d"
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
