import os

class TSSeg(object):
    def __init__(self):
        self.seg_cfgs = [
            "1conv1-64-19", "3conv1-64-19",
            "1conv2-64-19", "3conv2-64-19",
            "1conv3-64-19", "3conv3-64-19"
            ]
        self.basecmd = "python train/tssegtrain.py --task tsseg --seg-cfg %s --gpu %s --batch_size %d &"
    
    def args_gen(self, gpus):
        l = []
        count = 0

        for segcfg in self.seg_cfgs:
            gpu = gpus[count]
            batch_size = 2
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

gpus = ["3,0", "4,1", "5,0", "6,1"]
assign_run(TSSeg().command, gpus)