import numpy as np

# column vector
W = np.random.randn(5, 3)
Q, _ = np.linalg.qr(W)