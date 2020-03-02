import os

for i in range(10):
    os.system("python figure/qualitative_lsun.py %d" % (i + 1001))