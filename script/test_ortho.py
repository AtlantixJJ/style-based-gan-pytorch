import numpy as np
# W (1520, 16)
# column vector
W = np.random.randn(16, 1520, 1, 1)
W_ = W[:, :, 0, 0].transpose() # (1520, 16)
Q, R = np.linalg.qr(W)
I = np.ones(1520, 1520)
