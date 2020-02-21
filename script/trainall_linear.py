import os

#train_size = list(range(1, 4)); gpus = [0]
# 1, 2, 3, 4
train_size = [6, 8, 10, 1024, 40, 48, 64, 96, 128, 192, 256, 384, 512, 12, 16, 20, 24, 32]; gpus = [4,5,6,7]
#train_size = list(range(72, 97, 8)); gpus = [0, 1]
basecmd = "python script/linear_multiple_image.py --train-size %d --gpu %d --total-repeat 5"

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
