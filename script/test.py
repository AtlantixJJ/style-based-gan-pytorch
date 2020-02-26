from __future__ import print_function
import numpy as np
import pymp

a = np.random.rand(10000, 50)

#p_a = pymp.shared.array((10000, 50), dtype='float32')
#p_a[:] = a
with pymp.Parallel(4) as p:
    for index in p.range(0, 4):
        arr = a[index * 2500:(index + 1) * 2500]
        res = np.matmul(arr.transpose(), arr).sum()
        # The parallel print function takes care of asynchronous output.
        p.print('Yay! {} done!'.format(res))
for index in range(0, 4):
    arr = a[index * 2500:(index + 1) * 2500]
    res = np.matmul(arr.transpose(), arr).sum()
    # The parallel print function takes care of asynchronous output.
    p.print('Yay! {} done!'.format(res))