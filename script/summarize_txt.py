import sys, glob

files = glob.glob(f"{sys.argv[1]}/*.txt")
files.sort()
print(files)
lines = []
for f in files:
    with open(f, "r") as fp:
        lines.append(fp.read().strip())
print("\n".join(lines))