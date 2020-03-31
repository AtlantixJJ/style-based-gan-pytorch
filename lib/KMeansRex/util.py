import numpy as np
import KMeansRex
X = np.random.rand(1000,5)
Mu, Z = KMeansRex.RunKMeans(X, 3)
