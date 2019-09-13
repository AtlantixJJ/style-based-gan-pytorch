import os
dirs = [f for f in os.listdir("expr") if os.path.isdir("expr/" + f)]
dirs.sort()
print(dirs)
basecmd = "python monitor.py --model expr/%s --lerp 1 --task latest,evol,lerp"
for d in dirs:
  cmd = basecmd % d
  print(cmd)
  os.system(cmd)