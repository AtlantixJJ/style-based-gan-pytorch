import os
basecmd = "python tstrain.py --ma 1.0 --md 1.0 --mv 1.0 --att %d --att-mtd %s --gpu %s --batch_size %d &"
gpus = ["0,2", "1,3", "1,4", "0,5", "0,6"]
slots = [""] * len(gpus)
atts = [2, 4, 8]
att_mtds = ["sep-gencos1-ch", "sep-gen1-ch", "sep-gen1-elt",
            "sep-gencos2-ch", "sep-gen2-ch", "sep-gen2-elt",
            "uni-gencos2-ch", "uni-gencos1-ch"]


def args_gen(atts, att_mtds, gpus):
    l = []
    count = 0
    for att_mtd in att_mtds:
        for att in atts:
            gpu = gpus[count]
            if "sep" in att_mtd:
                if att == 2:
                    batch_size = 8
                elif att == 4:
                    batch_size = 6
                elif att == 8:
                    batch_size = 4
            else:
                batch_size = 8
            l.append((count, (att, att_mtd, gpu, batch_size)))
            count = (count + 1) % len(gpus)
    return l


for count, arg in args_gen(atts, att_mtds, gpus):
    cmd = basecmd % arg
    slots[count] += cmd + "& "

for s in slots:
    print(s[:-2])
    os.system(s[:-2])
