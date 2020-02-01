import sys
sys.path.insert(0, ".")
import utils
import numpy as np

filename = sys.argv[1]
dic = np.load(filename, allow_pickle=True)[()]
if "agreement" in filename:
    utils.format_agreement_result(dic)
else:
    utils.format_test_result(dic)

