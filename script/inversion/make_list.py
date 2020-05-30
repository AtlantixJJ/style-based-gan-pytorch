import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="")
parser.add_argument("--n-gpu", default=8, type=int)
parser.add_argument("--total", default=100, type=int)
args = parser.parse_args()

dir = args.dir
image_dir = "CelebA-HQ-img"
label_dir = "CelebAMask-HQ-mask-15"
images = glob.glob(f"{dir}/{image_dir}/*")
labels = glob.glob(f"{dir}/{label_dir}/*")
images.sort()
labels.sort()

n_gpu = args.n_gpu
n_total = min(args.total, len(images))
step = int(n_total / n_gpu)
for i in range(n_gpu):
    lines = zip(
        images[step * i : step * (i + 1)],
        labels[step * i : step * (i + 1)])
    with open(f"inversion_list_{i}.txt", "w") as f:
        f.write("\n".join([" ".join(l) for l in lines]))
