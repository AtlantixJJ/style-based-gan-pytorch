import os
cmds = [
    f"python figure/lsun_agreement.py --dir record/lsun --model stylegan_",
    f"python figure/lsun_agreement.py --dir record/lsun --model proggan",
    f"python figure/lsun_agreement.py --dir record/lsun --model stylegan2",]
for cmd in cmds:
    os.system(cmd)