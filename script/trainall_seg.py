import os, subprocess

basecmd = "python train/wgan.py --task wgan --gpu 0,1,2,3 --batch-size 256 --iter-num 100000 --imsize 64 --load "" --lr 0.0002 --dataset datasets/CelebAMask-HQ/CelebA-HQ-img-64"

basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg conv-16-1 --gpu 0 --batch-size 4 --iter-num 8000 --trace 1 --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt --expr record/celebahq"


class SECore(object):
    def __init__(self):
        self.last_only = [1, 0]
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
        ]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor %s --gpu %s --batch-size 1 --iter-num 10000 --last-only %d --expr record/celebahq%d"

    def args_gen(self, gpus):
        l = []
        count = 0
        for j in self.last_only:
            for i in range(len(self.extractors)):
                extractor = self.extractors[i]
                gpu = gpus[count]
                l.append((count, (extractor, gpu, j, j)))
                count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd


class SEL1Reg(SECore):
    def __init__(self):
        self.l1_reg = [1e-6, 1e-5, 1e-4, 1e-3]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --l1-reg %f --gpu %s --batch-size 1 --iter-num 5000 --last-only 0"

    def args_gen(self, gpus):
        l = []
        count = 0
        for i in range(len(self.l1_reg)):
            l1 = self.l1_reg[i]
            gpu = gpus[count]
            l.append((count, (l1, gpu)))
            count = (count + 1) % len(gpus)
        return l


class SESpherical(SECore):
    def __init__(self):
        self.last_only = [0, 1]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor spherical --gpu %s --batch-size 1 --iter-num 20000 --last-only %d --expr record/celebahq%d"

    def args_gen(self, gpus):
        l = []
        count = 0
        for j in self.last_only:
            gpu = gpus[count]
            l.append((count, (gpu, j, j)))
            count = (count + 1) % len(gpus)
        return l
    

def assign_run(command_generator, gpus, false_exec=False):
    slots = [""] * len(gpus)
    for index, cmd in command_generator(gpus):
        slots[index] += cmd + " && "
    for s in slots:
        print(s[:-2])
        if not false_exec:
            os.system(s[:-2])


def direct_run(gpus):
    commands = [
        "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --norm-reg 1.0 --gpu %s --batch-size 1 --iter-num 5000 --last-only 0",
        #"python train/extract_semantics.py --load checkpoint/bedroom_lsun_256x256_proggan.pth --model-name proggan --batch-size 1 --iter-num 30000 --last-only 0 --task bedroom --gpu %s",
        #"python train/extract_semantics.py --load checkpoint/bedroom_lsun_256x256_stylegan.pth --model-name stylegan --batch-size 1 --iter-num 30000 --last-only 0 --task bedroom --gpu %s"
        ]
    for i in range(len(commands)):
        index = i % len(gpus)
        gpu = gpus[index]
        c = commands[i]
        yield index, c % gpu


uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "jericho" in uname:
    gpus = ["0"]; assign_run(SECore().command, gpus)
    #gpus = ["0"]; assign_run(direct_run, gpus)
elif "instance" in uname:
    gpus = ["0"]; assign_run(SESpherical().command, gpus)
