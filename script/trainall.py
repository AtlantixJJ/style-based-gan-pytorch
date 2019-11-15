import os

class FixSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "1cas1-64-16",
            "1cas2-64-16",
            ]

        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --arch tfseg --gpu %s --batch_size 1 --iter-num 1000 --zero 1 &"
    
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
        self.seg_cfgs = ["3conv1-64-19", "3conv2-64-19"]
        self.basecmd = "python train/tssegtrain.py --task fixseg --seg-cfg %s --arch tfseg --gpu %s --batch_size 1 --iter-num 2000 &"

def assign_run(command_generator, gpus, false_exec=False):
    slots = [""] * len(gpus)
    for count, cmd in command_generator(gpus):
        slots[count] += cmd + "& "
    for s in slots:
        print(s[:-2])
        if not false_exec:
            os.system(s[:-2])

gpus = ["0", "1", "2", "3"]; assign_run(FixSeg().command, gpus)
#gpus = ["2,3"]; assign_run(TSSeg().command, gpus)