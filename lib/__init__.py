"""
Third party script. Sometimes modified to accomodate.
"""
import sys
sys.path.insert(0, ".")
import lib.face_parsing

import subprocess
uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "img14" not in uname and "Darwin" not in uname:
    import lib.op
else:
    print("!> StyleGAN2 not available")