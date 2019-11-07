import os

class TSSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "3res1-64-19",
            "3res2-64-19",
            "3conv1-64-19",
            "3conv2-64-19",
            "3conv2-32-19",
            "3conv1-32-19",
            "3res1-32-19",
            "3res2-32-19",
            ]
        self.basecmd = "python train/fixsegtrain.py --task fixseg --seg-cfg %s --gpu %s --batch_size %d --iter-num 2000 &"
    
    def args_gen(self, gpus):
        l = []
        count = 0

        for segcfg in self.seg_cfgs:
            gpu = gpus[count]
            batch_size = 1
            l.append((count, (segcfg, gpu, batch_size)))
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

gpus = ["0", "3"]
assign_run(TSSeg().command, gpus)