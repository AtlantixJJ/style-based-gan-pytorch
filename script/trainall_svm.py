import os

layer_index = [
    "3",
    "4",
    "5",
    "6",
    "3,4",
    "4,5",
    "5,6",
    "3,4,5",
    "4,5,6",
    "3,4,5,6"]

basecmd = "python script/svm_multiple_image.py --layer-index %s"

def command():
    for i, l in enumerate(layer_index):
        yield i, basecmd % l

slot = []
for i, c in command():
    slot.append(c)
cmd = " && ".join(slot)
print(cmd)
os.system(cmd)