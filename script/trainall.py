import os

class FixSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "conv-16-3",
            "conv-16-2",
            "conv-16-1",
            ]
        self.models = [
            "checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt",
            "checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt"
        ]
        self.output_dirs = [
            "celebahq",
            "ffhq"
        ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch_size 4 --iter-num 8000 --trace %d --load %s --expr %s"
        # evaluator: layer visualization is missing

    def args_gen(self, gpus):
        l = []
        count = 0
 
        for i in range(len(self.seg_cfgs)):
            segcfg = self.seg_cfgs[i]
            gpu = gpus[count]
            
            if "mul" in segcfg:
                trace = 1
            else:
                trace = 0
            
            for model, expr in zip(self.models, self.output_dirs):
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

class FixSegL1(object):
    def __init__(self):
        self.seg_cfgs = [
            "mul-16-l1",
            "mul-16-l1",
            "mul-16-l1"
            ]
        self.l1_coefs = [
            2e-4,
            1e-3,
            5e-3,
        ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch_size %d --iter-num %d --trace %d --reg-coef %f --load checkpoint/karras2019stylegan-celebahq-1024x1024.for_g_all.pt &"
    
    def args_gen(self, gpus):
        l = []
        count = 0
 
        for i in range(len(self.seg_cfgs)):
            segcfg = self.seg_cfgs[i]
            l1coef = self.l1_coefs[i]
            gpu = gpus[count]

            trace = batch_size = iter_num = 0
            if "mul" in segcfg:
                trace = 1
                batch_size = 8
                iter_num = 4000
            else:
                trace = 0
                batch_size = 4
                iter_num = 8000

            l.append((count, (segcfg, gpu, batch_size, iter_num, trace, l1coef)))
            count = (count + 1) % len(gpus)
        return l
    
    def command(self, gpus):
        for count, arg in self.args_gen(gpus):
            cmd = self.basecmd % arg
            yield count, cmd


def assign_run(command_generator, gpus, false_exec=False):
    slots = [""] * len(gpus)
    for index, cmd in command_generator(gpus):
        slots[index] += cmd + "& "
    for s in slots:
        print(s[:-2])
        if not false_exec:
            os.system(s[:-2])


def direct_run(gpus):
    commands = [
        "python train/fixsegtrain.py --task fixseg --seg-cfg mul-16 --arch tfseg --batch_size 1 --gpu %s --trace 1 &",
        #"python train/fixsegtrain.py --task fixsegsimple --seg-cfg mul-16 --arch simple --batch_size 1 --load expr/celeba_wgan_64/gen_iter_100000.model --imsize 64 --seg-net checkpoint/faceparse_unet_128.pth --gpu %s --trace 1 &",
        #"python train/simplesdtrain.py --task simplesd --seg-cfg mul-16 --arch simple --batch_size 64 --imsize 256 --load \"\" --disc-net \"\" --gpu %s --iter-num 100000 &",
        #"python train/simplesdtrain.py --task simplesd --seg-cfg mul-16 --seg 0 --arch simple --batch_size 64 --imsize 256 --load \"\" --disc-net \"\" --gpu %s --iter-num 100000 &",
        #"python train/segtrain.py --task simpleseg --arch simple --batch_size 64 --imsize 64 --load \"\" --disc-net \"\" --gpu %s &",
        #"python train/guidesdtrain.py --task simplesd --seg-cfg mul-16 --arch simple --batch_size 64 --imsize 64 --load \"\" --disc-net \"\" --gpu %s --guide norm &",
        #"python train/guidesdtrain.py --task simplesd --seg-cfg mul-16 --arch simple --batch_size 64 --imsize 64 --load \"\" --disc-net \"\" --gpu %s --guide delta &",
    ]
    for i in range(len(commands)):
        index = i % len(gpus)
        gpu = gpus[index]
        c = commands[i]
        yield index, c % gpu

#gpus = ["6", "7"]; assign_run(direct_run, gpus)
gpus = ["0", "1", "2", "3"]; assign_run(FixSeg().command, gpus)
#gpus = ["0", "1", "2"]; assign_run(FixSeg().command, gpus)