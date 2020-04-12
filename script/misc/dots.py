import os
import glob
files = glob.glob("./*.dot")
files.sort()
for f in files:
  x = f.replace("dot", "png")
  os.system(f"dot -o{x} -Tpng {f}")
