"""
Third party script. Sometimes modified to accomodate.
"""
import sys
sys.path.insert(0, ".")
import lib.face_parsing
try:
    import lib.op
except:
    print("!> StyleGAN2 not available")