import numpy as np

rng = np.random.RandomState(65537)
latent = rng.randn(256, 128)
np.save("datasets/simple_latent.npy", latent)