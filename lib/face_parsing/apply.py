import sys, os
sys.path.insert(0, ".")
import torch
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F
import unet
import argparse
from PIL import Image
import numpy as np
from lib.face_parsing.utils import tensor2label
from dataset import SimpleDataset
from tqdm import tqdm

def imread(fpath):
	with open(fpath, "rb") as fp:
		return np.asarray(Image.open(fp)).astype("float32")

def imwrite(fpath, img):
	# (-1, 1)
	with open(fpath, "wb") as fp:
		Image.fromarray(((img + 1) * 127.5).astype("uint8")).save(fp, fotmat="PNG")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoint/faceparse_unet.pth")
parser.add_argument("--file", default="")
parser.add_argument("--dir", default="")
args = parser.parse_args()

net = unet.unet()
state_dict = torch.load(args.model)
net.load_state_dict(state_dict)
net = net.cuda()

def preprocess(fpath):
	img = imread(fpath)
	img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
	img_t = (img_t - 127.5) / 255
	img_t = F.interpolate(img_t, (512, 512), mode='bilinear')
	return img_t

if args.file != "":
	img_t = preprocess(fpath)
	pred = net(img_t)
	imwrite("tmp.png", tensor2label(pred[0], 19).transpose(1, 2, 0))

if args.dir != "":
	T = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
	ds = SimpleDataset(args.dir, 512, T)
	dl = torch.utils.data.DataLoader(ds, batch_size=1)
	for i,img_t in tqdm(enumerate(dl)):
		pred = net(img_t.cuda())[0]
		# argmax and normalize to [0, 1]
		raw_label = pred.argmax(0, keepdim=True) / float(pred.shape[0])
		raw_image = torch.cat([raw_label, raw_label, raw_label], dim=0)
		raw_image = raw_image.cpu().numpy().transpose(1, 2, 0)
		label = tensor2label(pred, pred.shape[0]).transpose(1, 2, 0)
		imwrite("label/%05d.png" % i, raw_image)
		imwrite("label_viz/%05d.png" % i, label)
