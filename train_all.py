import os
basecmd = "python tstrain.py --ma 1.0 --md 1.0 --mv 1.0 --att %d --att-mtd %s --gpu %s --batch_size %d &"
gpus = ["0,2", "1,3", "1,4", "0,5", "0,6"]
slots = [""] * len(gpus)
atts = [2, 4, 8]
"""
att_mtds = ["sep-conv1-ch", "sep-conv1-elt", "sep-conv2-ch", "sep-conv2-elt",
            "sep-gencos1-ch", "sep-gen1-ch", "sep-gen1-elt",
            "sep-gencos2-ch", "sep-gen2-ch", "sep-gen2-elt"]
"""
att_mtds = ["sep-conv1-ch"]


def args_gen(atts, att_mtds, gpus):
    l = []
    count = 0

    for att in atts:
        for att_mtd in att_mtds:
            gpu = gpus[count]
            batch_size = 4
            l.append((count, (att, att_mtd, gpu, batch_size)))
            count = (count + 1) % len(gpus)
    return l


for count, arg in args_gen(atts, att_mtds, gpus):
    cmd = basecmd % arg
    slots[count] += cmd + "& "

for s in slots:
    print(s[:-2])
    os.system(s[:-2])
