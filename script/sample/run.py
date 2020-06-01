import os
import sys
import subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--task", default="")
parser.add_argument("--name", default="")
parser.add_argument("--extractor", default="")
parser.add_argument("--ds", default="")
args = parser.parse_args()



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


def baseline_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/baseline.py --outdir results/baseline_real --n-iter 1600 --n-total 8 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d"
    for i in range(8):
        idx = count % len(gpus)
        yield idx, basecmd % (i, i, gpus[idx])
        count += 1


def command_pggan_fewshot_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal_pggan.py --outdir results/sample_pggan_%d --n-iter 1000 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/svm_t%d_stylegan_linear_extractor_layer3,4,5,6,7.model --G checkpoint/face_celebahq_1024x1024_proggan.pth --resolution 512 --gpu %d"

    for t in [1, 2, 4, 8, 16]:
        for i in range(8):
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1


def command_sample_real(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --n-iter 1600 --n-total 16 --image ../datasets/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../datasets/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --gpu %d --method %s --outdir results/mask_sample_real_method"
    for i in range(10):
        for method in ["ML", "GL", "EL", "LL"]:
            idx = count % len(gpus)
            yield idx, basecmd % (i, i, gpus[idx], method)
            count += 1


def command_bedroom_stylegan_fewshot(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/bedroom_stylegan_sample_fewshot_%d --n-iter 3000 --n-total 8 --image datasets/Bedroom_StyleGAN_SV_full/image%d.png --label datasets/Bedroom_StyleGAN_SV_full/sv_label%d.npy --model results/svm_t%d_bedroom_lsun_stylegan_linear_extractor_layer2,3,4,5,6.model --G checkpoint/bedroom_lsun_256x256_stylegan.pth --resolution 256 --gpu %d --method LL"

    for i in range(20, 30):
        for t in [1, 2, 4, 8, 16]:
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1


def sample_fewshot_real_face(gpus):
    count = 0
    basecmd = "python script/sample/msreal.py --outdir results/face_stylegan_fewshot_real_%d --n-iter 1000 --n-total 8 --image ../data/CelebAMask-HQ/CelebA-HQ-img/%d.jpg --label ../data/CelebAMask-HQ/CelebAMask-HQ-mask-15/%d.png --model results/fewshot_svm/svm_t%d_face_celebahq_stylegan_layer2,3,4,5,6,7,8_linear_extractor.model --resolution 1024 --gpu %d"
    for t in [1, 2, 4, 8]:
        for i in range(10):
            idx = count % len(gpus)
            yield idx, basecmd % (t, i, i, t, gpus[idx])
            count += 1

if args.task == "0": # sample fewshot face
    command = sample_fewshot_real_face
    
elif args.task == "1": # sample fewshot uper
    ds = args.ds
    name = args.name # bedroom_lsun_stylegan

    def sample_fewshot_real_uper(gpus):
        count = 0
        model_path = "bedroom_lsun_256x256_stylegan.pth"
        resolution = 256
        l = ""
        if "church" in name:
            model_path = "church_lsun_256x256_stylegan2.pth"
        elif "cat" in name:
            model_path = "cat_lsun_256x256_stylegan2.pth"
        elif "car" in name:
            model_path = "car_lsun_512x384_stylegan2.pth"
            l = ",7"
            resolution = 512
        niter = 200
        basecmd = f"python script/sample/msreal.py --outdir results/{name}_fewshot_real_%d --n-iter {niter} --n-total 8 --image {ds}/image%d.png --label {ds}/sv_label%d.npy --model results/fewshot_svm/svm_t%d_{name}_layer2,3,4,5,6{l}_linear_extractor.model --G checkpoint/{model_path} --resolution {resolution} --gpu %d --method LL"

        for t in [8]:
            for i in range(10):
                idx = count % len(gpus)
                yield idx, basecmd % (t, i, i, t, gpus[idx])
                count += 1
    command = sample_fewshot_real_uper

elif args.task == "2": # sample uper
    ds = args.ds
    name = args.name # bedroom_lsun_stylegan
    extractor = args.extractor
    model_path = "bedroom_lsun_256x256_stylegan.pth"
    resolution = 256
    n_iter = 200
    if "church" in name:
        model_path = "church_lsun_256x256_stylegan2.pth"
    elif "cat" in name:
        model_path = "cat_lsun_256x256_stylegan2.pth"
    elif "car" in name:
        model_path = "car_lsun_512x384_stylegan2.pth"
        resolution = 512

    def command_sample_real_uper(gpus):
        count = 0
        basecmd = f"python script/sample/msreal.py --n-iter {n_iter} --n-total 8 --image {ds}/image%d.png --label {ds}/sv_label%d.npy --model {extractor} --G checkpoint/{model_path} --resolution {resolution} --gpu %d --method LL --outdir results/{name}_real"
        for i in range(10):
            idx = count % len(gpus)
            yield idx, basecmd % (i, i, gpus[idx])
            count += 1
    
    command = command_sample_real_uper


elif args.task == "3": # sample partial face 
    ds = "datasets/face_doodle"
    name = "face_celebahq_stylegan"
    def sample_fewshot_partial_face(gpus):
        count = 0
        basecmd = f"python script/sample/msreal.py --outdir results/{name}_fewshot_partial_%d --n-iter 200 --n-total 8 --label {ds}/%d.npy --model results/fewshot_svm/svm_t%d_face_celebahq_stylegan_layer2,3,4,5,6,7,8_linear_extractor.model --G checkpoint/face_celebahq_1024x1024_stylegan.pth --resolution 1024 --gpu %d --method %s"

        for t in [16, 1, 2, 4, 8]:
            for i in range(6):
                for method in ["LL", "ML"]:
                    idx = count % len(gpus)
                    yield idx, basecmd % (t, i, t, gpus[idx], method)
                    count += 1
    command = sample_fewshot_partial_face

gpus = [int(g) for g in args.gpu.split(',')]
slots = [[] for _ in gpus]
for i, c in command(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
