import os, subprocess

basecmd = "python train/wgan.py --task wgan --gpu 0,1,2,3 --batch-size 256 --iter-num 100000 --imsize 64 --load "" --lr 0.0002 --dataset datasets/CelebAMask-HQ/CelebA-HQ-img-64"

basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg conv-16-1 --gpu 0 --batch-size 4 --iter-num 8000 --trace 1 --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt --expr record/celebahq"

class FixSegCore(object):
    def __init__(self):
        self.segcfgs = [
            "conv-16-1",
            "conv-16-2",
            "conv-16-3"
        ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch-size 4 --iter-num 8000 --trace %s --expr record/celebahq"

    def args_gen(self, gpus):
        l = []
        count = 0
 
        for i in range(len(self.segcfgs)):
            segcfg = self.segcfgs[i]
            
            if "conv-16-1" in segcfg:
                trace = 1
            else:
                trace = 0
            
            gpu = gpus[count]
            l.append((count, (segcfg, gpu, trace)))
            count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd


class FixSegReg(FixSegCore):
    def __init__(self):
        self.segcfgs = [
            "conv-16-1_positive",
            "conv-16-1_positive_bias",
            "conv-16-1",
            "conv-16-1"
        ]
        self.reg_weights = [
            (-1, -1),
            (-1, -1),
            (1.0, -1),
            (-1, 1.0)
        ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch-size 4 --iter-num 8000 --trace %s --ortho-reg %f --positive-reg %f --expr record/celebahq"

    def args_gen(self, gpus):
        l = []
        count = 0
 
        for i in range(len(self.segcfgs)):
            segcfg = self.segcfgs[i]
            ortho_coef = self.reg_weights[i][0]
            positive_coef = self.reg_weights[i][1]
            
            if "conv-16-1" in segcfg:
                trace = 1
            else:
                trace = 0
            
            gpu = gpus[count]
            l.append((count, (segcfg, gpu, trace, ortho_coef, positive_coef)))
            count = (count + 1) % len(gpus)
        return l


class FixSegFull(object):
    def __init__(self):
        # loop 1
        self.seg_cfgs = [
            #"conv-16-3",
            #"conv-16-2",
            "conv-16-1",
            ]
        # loop 2
        self.models = [
            "checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt",
            #"checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt"
        ]
        self.output_dirs = [
            "record/celebahq",
            #"record/ffhq"
        ]
        # loop 3
        self.positive_regs = [
            0.1,
            1.0
        ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch-size 4 --iter-num 8000 --trace %d --load %s --expr %s --positive-reg %f"
        # evaluator: layer visualization is missing

    def args_gen(self, gpus):
        l = []
        count = 0
 
        for i in range(len(self.seg_cfgs)):
            segcfg = self.seg_cfgs[i]
            
            if "conv-16-1" in segcfg:
                trace = 1
            else:
                trace = 0
            
            for model, expr in zip(self.models, self.output_dirs):
                if "conv-16-1" in segcfg:
                    for ortho_reg in self.positive_regs:
                        gpu = gpus[count]
                        l.append((count,
                            (segcfg, gpu, trace, model, expr, ortho_reg # train
                            #gpu, expr, segcfg
                            ))) # eval
                        count = (count + 1) % len(gpus)
                else:
                    gpu = gpus[count]
                    l.append((count,
                        (segcfg, gpu, trace, model, expr, # train
                        #gpu, expr, segcfg
                        ))) # eval
                    count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd


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
        "python train/semantic_extractor.py --load checkpoint/bedroom_lsun_256x256_proggan.pth --model-name proggan --batch-size 2 --iter-num 16000 --task bedroom --gpu %s",
        "python train/semantic_extractor.py --load checkpoint/bedroom_lsun_256x256_stylegan.pth --model-name stylegan --batch-size 2 --iter-num 16000 --task bedroom --gpu %s"]
    for i in range(len(commands)):
        index = i % len(gpus)
        gpu = gpus[index]
        c = commands[i]
        yield index, c % gpu


uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "jericho" in uname:
    #gpus = ["0"]; assign_run(FixSegCore().command, gpus)
    gpus = ["0"]; assign_run(direct_run, gpus)
elif "instance" in uname:
    gpus = ["0", "1", "2", "3"]; assign_run(FixSegReg().command, gpus)
