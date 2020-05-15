import os

#train_size = list(range(1, 4)); gpus = [0]
# 1, 2, 3, 4
train_size = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 80, 96, 128, 256, 384, 512, 768, 1024]; gpus = [0, 1, 2, 3, 4, 5, 6, 7]
#train_size = list(range(72, 97, 8)); gpus = [0, 1]


def command_linear_multiple(gpus):
    count = 0
    basecmd = "python script/linear_multiple_image.py --train-size %d --gpu %d --repeat-idx %d"
    for ind in range(5):
        for j, ts in enumerate(train_size):
            idx = count % len(gpus)
            yield idx, basecmd % (ts, gpus[idx], ind)
            count += 1


def command_sv(gpus):
    count = 0
    basecmd = "python script/fewshot/sv_linear.py --single-class %d --data-dir datasets/SV_full"
    for c in range(2, 15):
        idx = count % len(gpus)
        yield idx, basecmd % c
        count += 1
gpus = [0]


def command_eval_trace(gpus):
    count = 0
    basecmd = "python script/analysis/eval_trace.py --n-segment 8 --segment %d --gpu %s --trace record/bce_kl/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos0.5_l1dev-1_l1unit-1/trace.npy"
    for idx, g in enumerate(gpus):
        yield idx, basecmd % (idx, g)
gpus = [0, 1, 2, 3, 4, 5, 6, 7]


slots = [[] for _ in gpus]
for i, c in command_eval_trace(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
