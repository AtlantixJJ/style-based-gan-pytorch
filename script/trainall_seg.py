import os
basecmd = "python train/fixsegtrain.py --task diffadainseg --seg-cfg %s --gpu %s --batch_size %d --iter-num 2000 &"
gpus = ["0", "1", "5", "6"]
slots = [""] * len(gpus)

seg_cfgs = ["1conv1-19", "3conv1-19",
            "1conv2-19", "3conv2-19",
            "1conv3-19", "3conv3-19"]

def args_gen(seg_cfgs, gpus):
    l = []
    count = 0

    for segcfg in seg_cfgs:
        gpu = gpus[count]
        batch_size = 8
        l.append((count, (segcfg, gpu, batch_size)))
        count = (count + 1) % len(gpus)
    return l


for count, arg in args_gen(seg_cfgs, gpus):
    cmd = basecmd % arg
    slots[count] += cmd + "& "

for s in slots:
    print(s[:-2])
    os.system(s[:-2])
