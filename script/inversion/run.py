import os
import sys

t = sys.argv[1]
f = "invert_semantics.py"
gpus = [0, 1, 2, 3]
mtd = "ML"
if t == "0":
    f = "invert_semantics.py"
    gpus = [0, 1, 2, 3]
    mtd = "ML"
elif t == "1":
    f = "invert_baseline.py"
    gpus = [4, 5, 6, 7]
    mtd = "ML"
elif t == "2":
    f = "invert_semantics.py"
    gpus = [0, 1, 2, 3]
    mtd = "EL"
elif t == "3":
    f = "invert_baseline.py"
    gpus = [4, 5, 6, 7]
    mtd = "EL"

def invert(gpus):
    count = 0
    basecmd = f"python script/inversion/{f} --imglist inversion_list_%d.txt --gpu %d --method {mtd}"
    for i in range(len(gpus)):
        idx = count % len(gpus)
        yield idx, basecmd % (i, gpus[idx])
        count += 1


slots = [[] for _ in gpus]
for i, c in invert(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
