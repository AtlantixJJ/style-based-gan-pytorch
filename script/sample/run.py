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


def command_sample_fewshot_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/mask_sample_fewshot_real_%d --n-iter 1600 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --resolution 512 --gpu %d"
    for i in range(100):
        for t in [1, 2, 4, 8]:
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1
gpus = [0, 1, 2, 3]

def command_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --n-iter 1600 --n-total 16 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d --method %s --outdir results/mask_sample_real_method"
    for i in range(10):
        for method in ["ML", "GL", "EL", "LL"]:
            idx = count % len(gpus)
            yield idx, basecmd % (i, i, gpus[idx], method)
            count += 1
gpus = [0]

def command_pggan_fewshot_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal_pggan.py --outdir results/sample_pggan_%d --n-iter 1000 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --G checkpoint/face_celebahq_1024x1024_proggan.pth --resolution 512 --gpu %d"

    for t in [1, 2, 4, 8, 16]:
        for i in range(8):
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1
gpus = [0, 1, 2, 3]

def baseline_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/baseline.py --outdir results/baseline_real --n-iter 1600 --n-total 8 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d"
    for i in range(8):
        idx = count % len(gpus)
        yield idx, basecmd % (i, i, gpus[idx])
        count += 1
gpus = [0]


slots = [[] for _ in gpus]
for i, c in command_pggan_fewshot_real(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
