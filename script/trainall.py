import os

class TSSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "3res1-64-16",
            "3res2-64-16",
            "3conv1-64-16",
            "3conv2-64-16",
            ]

        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --arch tfseg --gpu %s --batch_size 1 --iter-num 1000 &"
    
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

def assign_run(command_generator, gpus):
    slots = [""] * len(gpus)
    for count, cmd in command_generator(gpus):
        slots[count] += cmd + "& "
    for s in slots:
        print(s[:-2])
        os.system(s[:-2])

gpus = ["0", "1", "2", "3"]
assign_run(TSSeg().command, gpus)