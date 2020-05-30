import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="")
parser.add_argument("--label", default="")
parser.add_argument("--name", default="")
parser.add_argument("--n-gpu", default=1, type=int)
parser.add_argument("--total", default=100, type=int)
args = parser.parse_args()

images = glob.glob(f"{args.image}")
labels = glob.glob(f"{args.label}")
images.sort()
labels.sort()


n_gpu = args.n_gpu
n_total = min(args.total, len(images))
step = int(n_total / n_gpu)
for i in range(n_gpu):
    lines = zip(
        images[step * i : step * (i + 1)],
        labels[step * i : step * (i + 1)])
    with open(f"inversion_{args.name}_{i}.txt", "w") as f:
        f.write("\n".join([" ".join(l) for l in lines]))
