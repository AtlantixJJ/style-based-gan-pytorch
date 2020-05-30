import os
import sys


def command_svm_pca(gpus):
    count = 0
    basecmd = "python script/fewshot/svm_pca.py --train-size %d --pca-size 64"
    for ts in [1, 2, 4, 8, 16, 32]:
        idx = count % len(gpus)
        yield idx, basecmd % ts
        count += 1

def test_svm_pca(gpus):
    count = 0
    basecmd = "python script/fewshot/test_svm_model.py --model results/svm_pca_t%d_p32.model.npy --gpu %d"
    for ts in [1, 2, 4, 8, 16, 32]:
        idx = count % len(gpus)
        yield idx, basecmd % (ts, gpus[idx])
        count += 1

def command_linear_multiple(gpus):
    count = 0
    basecmd = "python script/linear_multiple_image.py --train-size %d --gpu %d --repeat-idx %d"
    for ind in range(5):
        for j, ts in enumerate(train_size):
            idx = count % len(gpus)
            yield idx, basecmd % (ts, gpus[idx], ind)
            count += 1

def command_eval_trace(gpus):
    count = 0
    basecmd = "python script/analysis/eval_trace.py --n-segment 8 --segment %d --gpu %s --trace record/bce_kl/celebahq_stylegan_unit_layer0,1,2,3,4,5,6,7,8_vbs1_l1-1_l1pos0.5_l1dev-1_l1unit-1/trace.npy"
    for idx, g in enumerate(gpus):
        yield idx, basecmd % (idx, g)

command = 0
gpus = [0]
if sys.argv[1] == "0":
    command = command_svm_pca
    gpus = [0, 0, 0, 0, 0]
elif sys.argv[1] == "1":
    command = test_svm_pca
    gpus = [0, 0, 0, 0, 0]
elif sys.argv[1] == "2":
    command = command_linear_multiple
    gpus = [0, 1, 2, 3, 4]
elif sys.argv[1] == "3":
    command = command_eval_trace
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
elif sys.argv[1] == "4":
    ds = sys.argv[2]
    name = sys.argv[3]
    # bedroom_lsun_stylegan, Bedroom_StyleGAN_SV_full
    # church_lsun_stylegan2, Church_StyleGAN2_SV_full

    def command_svm(gpus):
        count = 0
        basecmd = f"python script/fewshot/sv_linear.py --data-dir {ds} --name {name} --train-size %d --total-class 361"
        for ts in [1, 2, 4, 8, 16]:
            idx = count % len(gpus)
            yield idx, basecmd % ts
            count += 1
    gpus = [0]
    command = command_svm
    
slots = [[] for _ in gpus]
for i, c in command(gpus):
    slots[i].append(c)

for slot in slots:
    cmd = " && ".join(slot)
    print(cmd)
    os.system(cmd + " &")
