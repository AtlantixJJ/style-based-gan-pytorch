import os, subprocess

basecmd = "python train/wgan.py --task wgan --gpu 0,1,2,3 --batch-size 256 --iter-num 100000 --imsize 64 --load "" --lr 0.0002 --dataset datasets/CelebAMask-HQ/CelebA-HQ-img-64"

basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg conv-16-1 --gpu 0 --batch-size 4 --iter-num 8000 --trace 1 --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt --expr record/celebahq"


class SECore(object):
    def __init__(self):
        self.last_only = [1]
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
            "spherical"
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


class SEVBS(object):
    def __init__(self):
        self.vbs = [4, 16, 32, 64]
        self.extractors = [
            "linear",
            "unit"
        ]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor %s --gpu %s --batch-size 1 --iter-num %d --vbs %d --last-only 1 --expr record/vbs"

    def args_gen(self, gpus):
        l = []
        count = 0
        for j in self.vbs:
            for i in range(len(self.extractors)):
                extractor = self.extractors[i]
                gpu = gpus[count]
                l.append((count, (extractor, gpu, j * 1000, j)))
                count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd



class SEL1Reg(SECore):
    def __init__(self):
        self.l1_reg = ["7e-5", "8e-5", "9e-5"]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --l1-reg %s --gpu %s --batch-size 1 --iter-num 10000 --last-only 1 --expr record/l1/"

    def args_gen(self, gpus):
        l = []
        count = 0
        for i in range(len(self.l1_reg)):
            l1 = self.l1_reg[i]
            gpu = gpus[count]
            l.append((count, (gpu, l1, gpu)))
            count = (count + 1) % len(gpus)
        return l


class SELayers(SECore):
    def __init__(self):
        self.all_layers = "0,1,2,3,4,5,6,7,8"
        self.layer_num = 9
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --layers %s --gpu %s --batch-size 1 --iter-num 10000 --disp-iter 5000 --last-only 1 --expr record/layers/"

    # python script/monitor.py --task fast-celeba-agreement --model record/layers --recursive 1

    def args_gen(self, gpus):
        l = []
        count = 0
        for rev in [True]:
            for i in range(self.layer_num):
                layers = self.all_layers[2*i:]
                if rev:
                    if i < 5:
                        continue
                    layers = self.all_layers[:2*i-1]
                gpu = gpus[count]
                l.append((count, (gpu, layers, gpu)))
                count = (count + 1) % len(gpus)
        return l


class SEDiscLayers(SECore):
    def __init__(self):
        self.all_layers = "0,1,2,3,4,5,6,7"
        self.layer_num = 8
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics_disc.py --task celebahq --model-name stylegandisc --extractor linear --layers %s --gpu %s --batch-size 1 --iter-num 10000 --disp-iter 5000 --last-only 1 --expr record/disc_layers/ --imsize 1024 --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_d_basic.pt"

    # python script/monitor.py --task fast-celeba-agreement --model record/disc_layers --recursive 1

    def args_gen(self, gpus):
        l = []
        count = 0
        for rev in [False]:
            for i in range(self.layer_num):
                layers = self.all_layers[2*i:]
                if rev:
                    if i == 0:
                        continue
                    layers = self.all_layers[:2*i-1]
                else:
                    if i < 5:
                        continue
                gpu = gpus[count]
                l.append((count, (gpu, layers, gpu)))
                count = (count + 1) % len(gpus)
        return l


class SESpherical(SECore):
    def __init__(self):
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor spherical --gpu %s --batch-size 1 --iter-num 10000 --last-only 1 --expr record/celebahq1"

    def args_gen(self, gpus):
        l = []
        count = 0
        gpu = gpus[count]
        l.append((count, (gpu,)))
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
        ## unit
        #"python train/extract_semantics.py --task celebahq --model-name stylegan --extractor unit --gpu %s --batch-size 1 --iter-num 10000 --last-only 1",
        ## unit normalized
        #"python train/extract_semantics.py --task celebahq --model-name stylegan --extractor unitnorm --gpu %s --batch-size 1 --iter-num 10000 --last-only 1",
        ## continuous,
        "python train/extract_semantics_continuous.py --task celebahq --model-name stylegan --extractor unit --gpu %s --batch-size 1 --iter-num 32000 --last-only 1",
        ## church stylegan2
        #"python train/extract_semantics.py --load checkpoint/church_lsun_256x256_stylegan2.pth --model-name stylegan2 --batch-size 1 --iter-num 30000 --last-only 0 --task church --gpu %s",
        ## church prog
        #"python train/extract_semantics.py --load checkpoint/church_lsun_256x256_proggan.pth --model-name proggan --batch-size 1 --iter-num 30000 --last-only 0 --task church --gpu %s",
        ## bedroom prog
        #"python train/extract_semantics.py --load checkpoint/bedroom_lsun_256x256_proggan.pth --model-name proggan --batch-size 1 --iter-num 30000 --last-only 0 --task bedroom --gpu %s",
        ## bedroom stylegan
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
    #gpus = ["0"]; assign_run(SECore().command, gpus)
    #gpus = ["0"]; assign_run(SEL1Reg().command, gpus)
    #gpus = ["0"]; assign_run(direct_run, gpus)
    #gpus = ["0"]; assign_run(SEDiscLayers().command, gpus)
    gpus = ["0"]; assign_run(SEVBS().command, gpus)
elif "instance" in uname:
    gpus = ["0"]; assign_run(SESpherical().command, gpus)
else:
    gpus = ["4", "5", "6", "7"]; assign_run(SEL1Reg().command, gpus)
