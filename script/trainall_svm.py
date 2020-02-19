import os

train_size = [1, 2, 3, 4, 5, 6]

gpus = [0]
basecmd = "python script/svm_multiple_image.py --layer-index 2,3,4,5,6 --train-size %d --gpu %d"

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