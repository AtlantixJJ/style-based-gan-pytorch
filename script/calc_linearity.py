import sys, argparse, torch, glob, os
sys.path.insert(0, ".")
import numpy as np
import model, fid, utils
from lib.face_parsing.unet import unet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="../datasets/CelebAMask-HQ/CelebA-HQ-img-128")
parser.add_argument("--imsize", default=128, type=int)
parser.add_argument("--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument("--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument("--recursive", default=1, type=int)
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

ind = args.model.rfind("/")
model_name = args.model[ind + 1:]
device = "cuda"

if args.recursive == 1:
    start = 0
    if os.path.exists(f"record/{model_name}_linearity.txt"):
        with open(f"record/{model_name}_linearity.txt", "r") as f:
            lines = f.readlines()
        fids = [float(l.strip()) for l in lines]
        start = len(fids)
    # This is root, run for all the expr directory
    model_files = glob.glob(args.model + "/*.model")
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
    model_files = model_files[start:]
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, model in enumerate(model_files):
        basecmd = "python script/calc_fid_simple.py --dataset %s --model %s --gpu %s --recursive 0"
        basecmd = basecmd % (args.dataset, model, gpus[i % len(gpus)])
        slots[i % len(gpus)].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)


#generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
upsample = int(np.log2(args.imsize // 4))
generator = model.simple.Generator(upsample=upsample)
missed = generator.load_state_dict(torch.load(args.model), strict=False)
print(missed)
generator.to(device)

state_dict = torch.load(args.external_model, map_location='cpu')
faceparser = unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()

evaluator = utils.LinearityEvaluator(faceparser)
evaluator(generator)