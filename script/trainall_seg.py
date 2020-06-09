import os, subprocess

basecmd = "python train/wgan.py --task wgan --gpu 0,1,2,3 --batch-size 256 --iter-num 100000 --imsize 64 --load "" --lr 0.0002 --dataset datasets/CelebAMask-HQ/CelebA-HQ-img-64"

class SECore(object):
    def __init__(self):
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
            "spherical",
            "unit",
            "unitnorm"
        ]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor %s --gpu %s --expr record/celebahq1"

    def args_gen(self, gpus):
        l = []
        count = 0
        for i in range(len(self.extractors)):
            extractor = self.extractors[i]
            gpu = gpus[count]
            l.append((count, (extractor, gpu)))
            count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd


class SEBias(object):
    def __init__(self):
        self.layers = ["0,1,2,3,4,5,6,7,8", "3,4,5,6,7"]
        self.cmd1 = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --use-bias 1 --layers %s --gpu %s --expr record/bias"
        self.cmd2 = "python train/extract_semantics.py --task ffhq --model-name stylegan2 --extractor linear --layers %s --gpu %s --use-bias 1 --expr record/bias --load checkpoint/face_ffhq_1024x1024_stylegan2.pth"
    
    def command(self, gpus):
        l = []
        count = 0
        for cmd in [self.cmd2]:
            for layer in self.layers:
                gpu = gpus[count]
                count = (count + 1) % len(gpus)
                yield count, cmd % (layer, gpu)


class SEMix(SECore):
    def __init__(self):
        self.lams = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.extractors = [
            "linear",
            "nonlinear",
            "unit"]
        self.basecmd = "python train/es_mix.py --task celebahq --model-name stylegan --extractor %s --gpu %s --last-only 1 --l1-pos-reg %f --expr record/bce_kl"

    def args_gen(self, gpus):
        l = []
        count = 0
        for lam in self.lams:
            for i in range(len(self.extractors)):
                extractor = self.extractors[i]
                gpu = gpus[count]
                l.append((count, (extractor, gpu, lam)))
                count = (count + 1) % len(gpus)
        return l


class SECore2(SECore):
    def __init__(self):
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
            "spherical",
            "unit",
            "unitnorm"
        ]
        self.basecmd = "python train/extract_semantics.py --task ffhq --model-name stylegan2 --extractor %s --gpu %s --expr record/ffhq --load checkpoint/face_ffhq_1024x1024_stylegan2.pth"


class SEPGAN(SECore):
    def __init__(self):
        self.task = [
            #"bedroom",
            #"celebahq"
            "church",
            "conferenceroom",
            "diningroom",
            "kitchen",
            "livingroom",
            "restaurant"
            ]
        self.model = [
            #"bedroom_lsun_256x256_proggan.pth",
            #"face_celebahq_1024x1024_proggan.pth"
            "church_lsun_256x256_proggan.pth",
            "conferenceroom_lsun_256x256_proggan.pth",
            "diningroom_lsun_256x256_proggan.pth",
            "kitchen_lsun_256x256_proggan.pth",
            "livingroom_lsun_256x256_proggan.pth",
            "restaurant_lsun_256x256_proggan.pth",
            ]
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
            "unit",
            "unitnorm",
            "spherical"
        ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task %s --model-name proggan --extractor %s --gpu 0 --expr record/lsun --iter-num %d --load checkpoint/%s"

    def args_gen(self, gpus):
        l = []
        count = 0
        n_iter = 30000
        for i in range(len(self.extractors)):
            for t, m in zip(self.task, self.model):
                if self.task in ["celebahq", "ffhq"]:
                    n_iter = 10000
                extractor = self.extractors[i]
                gpu = gpus[count]
                l.append((count, (gpu, t, extractor, n_iter, m)))
                count = (count + 1) % len(gpus)
        return l


class SEStyleGAN(SEPGAN):
    def __init__(self):
        self.task = [
            "bedroom"]
        self.model = [
            "bedroom_lsun_256x256_stylegan.pth"]
        self.extractors = [
            "linear",
            "nonlinear",
            "generative",
            "unit",
            "unitnorm",
            "spherical"
        ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task %s --model-name stylegan --extractor %s --gpu 0 --expr record/lsun --iter-num %d --load checkpoint/%s"


class SEStyleGAN2(SEPGAN):
    def __init__(self):
        self.task = [
            "ffhq",
            #"church",
            ]
        self.model = [
            "face_ffhq_1024x1024_stylegan2.pth",
            #"church_lsun_256x256_stylegan2.pth"
            ]
        self.extractors = [
            #"linear",
            #"nonlinear",
            #"generative",
            #"unit",
            "unitnorm",
            "spherical"
        ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task %s --model-name stylegan2 --extractor %s --gpu 0 --expr record/lsun --iter-num %d --load checkpoint/%s"

class SEBCE(SECore):
    def __init__(self, extractors):
        self.extractors = extractors
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor %s --gpu %s --loss BCE --expr record/bce"

    def args_gen(self, gpus):
        l = []
        count = 0
        for i in range(len(self.extractors)):
            extractor = self.extractors[i]
            gpu = gpus[count]
            l.append((count, (extractor, gpu)))
            count = (count + 1) % len(gpus)
        return l


class SEVBS2(SECore):
    def __init__(self, extractors):
        self.vbs = [1, 4, 16, 32, 64]
        self.extractors = extractors
        self.basecmd = "python train/extract_semantics.py --task ffhq --model-name stylegan2 --extractor %s --gpu %s --iter-num %d --vbs %d --load checkpoint/face_ffhq_1024x1024_stylegan2.pth --expr record/vbs"

    def args_gen(self, gpus):
        l = []
        count = 0
        for j in self.vbs:
            for i in range(len(self.extractors)):
                extractor = self.extractors[i]
                gpu = gpus[count]
                l.append((count, (extractor, gpu, 32000, j)))
                count = (count + 1) % len(gpus)
        return l


class SEL1Reg(SECore):
    def __init__(self):
        self.l1_reg = [
            "0.001", "0.0001", "0.00001", "0.000001",
            #"0.009","0.008", "0.007", "0.006",
            #"0.005","0.004", "0.003", "0.002",
            #"0.0008", "0.0006", "0.0004", "0.0002"
            ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --l1-reg %s --gpu %s --loss BCE --expr record/l1_bce/"

    def args_gen(self, gpus):
        l = []
        count = 0
        for i in range(len(self.l1_reg)):
            l1 = self.l1_reg[i]
            gpu = gpus[count]
            l.append((count, (gpu, l1, gpu)))
            count = (count + 1) % len(gpus)
        return l


class SEL1PosReg(SEL1Reg):
    def __init__(self):
        self.l1_reg = [
            "0.01", "0.001", "0.0001", "0.00001"
            #"0.0001", "0.00001",
            #"0.009","0.008", "0.007", "0.006",
            #"0.005","0.004", "0.003", "0.002",
            #"0.0008", "0.0006", "0.0004", "0.0002"
            ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --l1-pos-reg %s --gpu %s --loss BCE --expr record/l1_pos_bce/"


class SEL1DEV(SEL1Reg):
    def __init__(self):
        self.l1_reg = [
            #"0.01", "0.001", "0.0001", "0.00001",
            "0.000001",
            #"0.009","0.008", "0.007", "0.006",
            #"0.005","0.004", "0.003", "0.002",
            #"0.0008", "0.0006", "0.0004", "0.0002"
            ]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --l1-stddev %s --gpu %s --loss BCE --expr record/l1_bce_stddev/"


class SELayers(SECore):
    def __init__(self):
        self.all_layers = ["0,1,2,3,4,5,6,7,8", "1,2,3,4,5,6,7,8", "2,3,4,5,6,7,8", "3,4,5,6,7,8", "3,4,5,6,7"]
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --task celebahq --model-name stylegan --extractor linear --layers %s --gpu %s --expr record/layers/"

    def args_gen(self, gpus):
        l = []
        count = 0
        for rev in [False, True]:
            for layers in self.all_layers:
                gpu = gpus[count]
                l.append((count, (gpu, layers, gpu)))
                count = (count + 1) % len(gpus)
        return l


class SEDiscLayers(SECore):
    def __init__(self):
        self.all_layers = "0,1,2,3,4,5,6,7"
        self.layer_num = 8
        self.basecmd = "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics_disc.py --task celebahq --model-name stylegandisc --extractor linear --layers %s --gpu %s --last-only 1 --expr record/disc_layers/ --imsize 1024 --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_d_basic.pt"

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


class SESGD(SECore):
    def __init__(self):
        self.extractors = ["linear", "unit", "nonlinear"]
        self.basecmd = "python train/extract_semantics.py --task celebahq --model-name stylegan --extractor %s --optim sgd --lr 0.01 --gpu %s --loss BCE --expr record/sgd_bce/"

class SESGD2(SECore):
    def __init__(self):
        self.extractors = ["linear", "unit", "nonlinear"]
        self.basecmd = "python train/extract_semantics.py --task ffhq --model-name stylegan2 --extractor %s --gpu %s --expr record/ffhq_sgd_bce --load checkpoint/face_ffhq_1024x1024_stylegan2.pth"

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
        #"python train/extract_semantics.py --task celebahq --model-name stylegan --extractor unit --gpu %s --last-only 1",
        ## unit normalized
        #"python train/extract_semantics.py --task celebahq --model-name stylegan --extractor unitnorm --gpu %s --last-only 1",
        ## continuous,
        #"python train/extract_semantics_continuous.py --task celebahq --model-name stylegan --extractor unit --gpu %s --iter-num 32000 --last-only 1",
        ## face stylegan2
        #"python train/extract_semantics_continuous.py --load checkpoint/face_ffhq_1024x1024_stylegan2.pth --model-name stylegan2 --iter-num 32000 --last-only 0 --task ffhq --gpu %s",
        ## church stylegan2
        "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --load checkpoint/church_lsun_256x256_stylegan2.pth --model-name stylegan2 --iter-num 30000 --task church --gpu 0",
        ## cat stylegan2
        "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --load checkpoint/cat_lsun_256x256_stylegan2.pth --model-name stylegan2 --iter-num 30000 --task cat --gpu 0",
        ## car stylegan2
        "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --load checkpoint/car_lsun_512x384_stylegan2.pth --model-name stylegan2 --iter-num 30000 --task car --gpu 0",
        ## horse stylegan2
        "CUDA_VISIBLE_DEVICES=%s python train/extract_semantics.py --load checkpoint/horse_lsun_256x256_stylegan2.pth --model-name stylegan2 --iter-num 30000 --task horse --gpu 0",
        ## church prog
        #"python train/extract_semantics.py --load checkpoint/church_lsun_256x256_proggan.pth --model-name proggan --iter-num 30000 --task church --gpu %s",
        ## bedroom prog
        #"python train/extract_semantics.py --load checkpoint/bedroom_lsun_256x256_proggan.pth --model-name proggan --iter-num 30000 --task bedroom --gpu %s",
        ## bedroom stylegan
        #"python train/extract_semantics.py --load checkpoint/bedroom_lsun_256x256_stylegan.pth --model-name stylegan --iter-num 30000 --task bedroom --gpu %s"
        ]
    for i in range(len(commands)):
        index = i % len(gpus)
        gpu = gpus[index]
        c = commands[i]
        yield index, c % gpu


uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "jericho" in uname:
    gpus = ["0"]; assign_run(SEStyleGAN2().command, gpus)
    #gpus = ["0"]; assign_run(SEPGAN().command, gpus)
    #gpus = ["0"]; assign_run(SECore2().command, gpus)
    #gpus = ["0"]; assign_run(direct_run, gpus)
    #gpus = ["0"]; assign_run(SEL1PosReg().command, gpus)
    #gpus = ["0"]; assign_run(SEBias().command, gpus)
    #gpus = ["0"]; assign_run(SEL1DEV().command, gpus)
    #gpus = ["0"]; assign_run(SEBCE(["linear"]).command, gpus)
    #gpus = ["0"]; assign_run(SEBCE(["unit"]).command, gpus)
elif "instance" in uname:
    gpus = ["0", "1", "2", "3"]; assign_run(SEStyleGAN2().command, gpus)
else:
    gpus = ["0", "1", "2", "3"]; assign_run(SEPGAN().command, gpus)
    gpus = ["5", "6", "7"]; assign_run(SEStyleGAN().command, gpus)
    
