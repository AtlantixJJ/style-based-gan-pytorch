import numpy as np
import glob
import os

rng = np.random.RandomState(1)
files = os.listdir("dlatent")
files.sort()

TRAIN_SIZE = 500
TEST_SIZE = len(files) - TRAIN_SIZE

train_files = rng.choice(files, TRAIN_SIZE)
test_files = [f for f in files if f not in train_files]

for f in train_files:
  os.system(f"cp dlatent/{f} dlatent_train/{f}")
  os.system(f"cp noise/{f} noise_train/{f}")

for f in test_files:
  os.system(f"cp dlatent/{f} dlatent_test/{f}")
  os.system(f"cp noise/{f} noise_test/{f}")