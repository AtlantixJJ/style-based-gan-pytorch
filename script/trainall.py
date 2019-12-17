import os

class FixSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "1cat1-64-16",
            "1mul1-64-16"
            ]

        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --arch tfseg --gpu %s --batch_size 1 --iter-num 5000 &"
    
    def args_gen(self, gpus):
        l = []
        count = 0

        for segcfg in self.seg_cfgs:
            gpu = gpus[count]
            l.append((count, (segcfg, gpu)))
            count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd

class TSSeg(FixSeg):
    def __init__(self):
        self.seg_cfgs = [
            "1cat1-64-16",
            "3cat1-64-16",
            #"1cas1-64-16",
            #"1gen1-64-16"
        ]
        self.basecmd = "python train/tssegtrain.py --task fixseg --seg-cfg %s --arch tfseg --gpu %s --batch_size 1 --iter-num 10000 &"

def assign_run(command_generator, gpus, false_exec=False):
    slots = [""] * len(gpus)
    for count, cmd in command_generator(gpus):
        slots[count] += cmd + "& "
    for s in slots:
        print(s[:-2])
        if not false_exec:
            os.system(s[:-2])

def direct_run(gpus):
    commands = [
        "python train/fixsegtrain.py --task fixseg --seg-cfg 1mul1-64-16 --arch tfseg --batch_size 1 --gpu %s &",
        "python train/fixsegtrain.py --task fixsegsimple --seg-cfg 1mul1-64-16 --arch simple --batch_size 1 --load expr/celeba_wgan_64/gen_iter_100000.model --imsize 64 --seg-net checkpoint/faceparse_unet_128.pth --gpu %s &",
        "python train/simplesdtrain.py --task simplesd --seg-cfg 1mul1-64-16 --arch simple --batch_size 64 --imsize 64 --load "" --disc-net "" --gpu %s &",
    ]
    for i, (c, gpu) in enumerate(zip(commands, gpus)):
        yield i, c % gpu

gpus = ["4", "5", "6", "7"]; assign_run(direct_run, gpus)
#gpus = ["4", "5", "6", "7"]; assign_run(FixSeg().command, gpus)
#gpus = ["0,2", "0,3"]; assign_run(TSSeg().command, gpus)