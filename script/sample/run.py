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


def command_fewshot(gpus):
    count = 0
    basecmd = "python script/sample/ms.py --kl-coef 0 --n-iter 1000 --seed %d --gpu %d --outdir results/mask_fewshot_%d --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --n-total 25"
    for t in [4, 8]:
        for s in [65537, 1314, 1973, 3771]:
            idx = count % len(gpus)
            yield idx, basecmd % (s, gpus[idx], t, t)
            count += 1
gpus = [0]

def command_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --n-iter 1600 --n-total 64 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d"
    for i in range(100):
        idx = count % len(gpus)
        yield idx, basecmd % (i, i, gpus[idx])
        count += 1
gpus = [0, 1, 2, 3]

def command_sample_fewshot_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/mask_sample_fewshot_real_%d --n-iter 1600 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --resolution 512 --gpu %d"
    for i in range(100):
        for t in [1, 2, 4, 8]:
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1
gpus = [0, 1, 2, 3]

slots = [[] for _ in gpus]
for i, c in command_sample_fewshot_real(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
