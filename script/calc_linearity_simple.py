"""
python script/calc_linearity_simple.py --imsize 64 --model expr/celeba_eyeg_wgan64/
"""
import sys, argparse, torch, glob, os
sys.path.insert(0, ".")
import numpy as np
import model, fid, utils, evaluate
from lib.face_parsing.unet import unet

parser = argparse.ArgumentParser()
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
    # This is root, run for all the expr directory
    model_files = glob.glob(args.model + "/*.model")
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, model in enumerate(model_files):
        basecmd = "python script/calc_linearity_simple.py --imsize %d --model %s --gpu %s --recursive 0"
        basecmd = basecmd % (args.imsize, model, gpus[i % len(gpus)])
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
train_size = 512
if "128" in args.external_model:
    train_size = 128
faceparser = unet(train_size=train_size)
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()

mapid = utils.CelebAIDMap().mapid

def external_model(x):
    return mapid(faceparser(x).argmax(1))

evaluator = evaluate.LinearityEvaluator(external_model, N=1000, latent_dim=128)
iou_std = evaluator(generator, model_name)
with open(f"record/{model_name}_linearity_ioustd.txt", "w") as f:
    f.write(f"{model_name} {iou_std}\n")