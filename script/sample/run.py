import os
import sys
import subprocess 

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

def command_fewshot(gpus):
    count = 0
    basecmd = "python script/sample/ms.py --kl-coef 0 --n-iter 1000 --seed %d --gpu %d --outdir results/mask_fewshot_%d --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --n-total 25"
    for t in [4, 8]:
        for s in [65537, 1314, 1973, 3771]:
            idx = count % len(gpus)
            yield idx, basecmd % (s, gpus[idx], t, t)
            count += 1

def command_sample_real_face(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --n-iter 1600 --n-total 16 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d --method %s --outdir results/mask_sample_real_method"
    for i in range(10):
        for method in ["ML", "GL", "EL", "LL"]:
            idx = count % len(gpus)
            yield idx, basecmd % (i, i, gpus[idx], method)
            count += 1
gpus = [0]

def baseline_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/baseline.py --outdir results/baseline_real --n-iter 1600 --n-total 8 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d"
    for i in range(8):
        idx = count % len(gpus)
        yield idx, basecmd % (i, i, gpus[idx])
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


def command_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --n-iter 1600 --n-total 16 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d --method %s --outdir results/mask_sample_real_method"
    for i in range(10):
        for method in ["ML", "GL", "EL", "LL"]:
            idx = count % len(gpus)
            yield idx, basecmd % (i, i, gpus[idx], method)
            count += 1
gpus = [0]

def command_bedroom_stylegan_fewshot(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/bedroom_stylegan_sample_fewshot_%d --n-iter 3000 --n-total 8 --image datasets/Bedroom_StyleGAN_SV_full/image%d.png --label datasets/Bedroom_StyleGAN_SV_full/sv_label%d.npy --model results/svm_t%d_bedroom_lsun_stylegan_linear_extractor_layer2,3,4,5,6.model --G checkpoint/bedroom_lsun_256x256_stylegan.pth --resolution 256 --gpu %d --method LL"

    for i in range(20, 30):
        for t in [1, 2, 4, 8, 16]:
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1
gpus = [0, 1, 2, 3, 5]


def sample_fewshot_real_face(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/face_stylegan_fewshot_real_%d --n-iter 1000 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/fewshot_svm/svm_t%d_face_celebahq_stylegan_layer2,3,4,5,6,7,8_linear_extractor.model --resolution 1024 --gpu %d"
    for t in [1, 2, 4, 8]:
        for i in range(10):
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1

gpus = [0]
if sys.argv[1] == "0": # sample fewshot face
    command = sample_fewshot_real_face
    gpus = [2, 3]
    
elif sys.argv[1] == "1": # sample fewshot uper
    ds = sys.argv[2]
    name = sys.argv[3] # bedroom_lsun_stylegan

    def sample_fewshot_real_uper(gpus):
        count = 0
        model_path = "bedroom_lsun_256x256_stylegan.pth"
        resolution = 256
        if "church" in name:
            model_path = "church_lsun_256x256_stylegan2.pth"
        elif "cat" in name:
            model_path = "cat_lsun_256x256_stylegan2.pth"
        elif "car" in name:
            model_path = "car_lsun_512x384_stylegan2.pth"
            resolution = 512

        basecmd = f"python script/sample/msreal.py --outdir results/{name}_fewshot_real_%d --n-iter 3000 --n-total 8 --image {ds}/image%d.png --label {ds}/sv_label%d.npy --model results/fewshot_svm/svm_t%d_{name}_layer2,3,4,5,6_linear_extractor.model --G checkpoint/{model_path} --resolution {resolution} --gpu %d --method LL"

        for t in [1, 2, 4, 8, 16]:
            for i in range(10):
                idx = count % len(gpus)
                yield idx, basecmd % (t, i, i, t, gpus[idx])
                count += 1
    gpus = [6, 7]
    command = sample_fewshot_real_uper

elif sys.argv[1] == "2": # sample uper
    ds = sys.argv[2]
    name = sys.argv[3] # bedroom_lsun_stylegan
    extractor = sys.argv[4]
    model_path = "bedroom_lsun_256x256_stylegan.pth"
    resolution = 256
    if "church" in name:
        model_path = "church_lsun_256x256_stylegan2.pth"
    elif "cat" in name:
        model_path = "cat_lsun_256x256_stylegan2.pth"
    elif "car" in name:
        model_path = "car_lsun_512x384_stylegan2.pth"
        resolution = 512

    def command_sample_real_uper(gpus):
        count = 0
        basecmd = f"python script/sample/msreal.py --n-iter 3000 --n-total 8 --image {ds}/image%d.png --label {ds}/sv_label%d.npy --model {extractor} --G checkpoint/{model_path} --resolution {resolution} --gpu %d --method %s"
        for i in range(10):
            for method in ["EL", "LL"]:
                idx = count % len(gpus)
                yield idx, basecmd % (i, i, gpus[idx], method)
                count += 1
    
    command = command_sample_real_uper
    gpus = [1, 3]




uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "instance" in uname:
    gpus = [0]

slots = [[] for _ in gpus]
for i, c in command(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
