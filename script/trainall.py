import os

class FixSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            #"mul-16-l1_nearest_sl0",
            #"mul-16-none_nearest_sl0",
            #"mul-16-l1_bilinear_sl0",
            #"mul-16-none_bilinear_sl0",
            #"mul-16-l1_nearest_sl1",
            #"mul-16-none_nearest_sl1",
            "mul-16-l1_bilinear_sl1", # stuck at here
            "mul-16-none_bilinear_sl1",
            "mul-16-l1_nearest_sl2",
            "mul-16-none_nearest_sl2",
            "mul-16-l1_bilinear_sl2",
            "mul-16-none_bilinear_sl2",
            ## complex
            #"gen-16-bilinear",
            #"gen-16-nearest",
            #"cas-16-bilinear",
            #"cas-16-nearest"
            ]

        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch_size %d --iter-num %d --trace %d &"
    
    def args_gen(self, gpus):
        l = []
        count = 0
 
        for segcfg in self.seg_cfgs:
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

            l.append((count, (segcfg, gpu, batch_size, iter_num, trace)))
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
#gpus = ["4", "5", "6", "7"]; assign_run(FixSeg().command, gpus)
gpus = ["1", "7"]; assign_run(FixSeg().command, gpus)