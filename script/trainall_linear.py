import os

train_size = list(range(12)); gpus = [0]
#train_size = list(range(12, 101, 4)); gpus = [4,5,6,7]

basecmd = "python script/linear_multiple_image.py --train-size %d --gpu %d"

def command(gpus):
    count = 0
    for j, ts in enumerate(train_size):
        idx = count % len(gpus)
        yield idx, basecmd % (ts, gpus[idx])
        count += 1

slots = [[] for _ in gpus]
for i, c in command(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")