"""
python script/calc_linearity_simple.py --model expr/celeba_hat_wgan128/ --recursive 0
python script/calc_linearity_simple.py --model expr/celeba_eyeg_wgan128/ --recursive 0
python script/calc_linearity_simple.py --imsize 64 --model expr/celeba_wgan64/
python script/calc_linearity_simple.py --imsize 64 --model expr/celeba_eyeg_wgan64/
python script/calc_linearity_simple.py --imsize 64 --model expr/celeba_eyeg_wgan64/gen_iter_001000.model --recursive 0
"""
import sys, os, argparse
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--imsize", default=128, type=int)
parser.add_argument("--model", default="expr/wgan128")
parser.add_argument("--external-model", default="checkpoint/faceparse_unet_512.pth")
parser.add_argument("--recursive", default=1, type=int)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--last-only", default=1, type=int)
parser.add_argument("--gpu", default="0")
parser.add_argument("--train-iter", default=1000, type=int)
parser.add_argument("--test-size", default=256, type=int)
parser.add_argument("--test-dir", default="datasets/Synthesized_test")
args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = "cuda"

import torch, glob
import numpy as np
import model, fid, utils, evaluate, dataset
from lib.face_parsing.unet import unet

if args.recursive == 1:
    # This is root, run for all the expr directory
    model_files = glob.glob(args.model + "/*.model")
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
    model_files = model_files[args.start:]
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, model in enumerate(model_files):
        gpu_idx = i % len(gpus)
        basecmd = "python script/calc_linearity_simple.py --imsize %d --model %s  --last-only %d --gpu %s --recursive 0"
        basecmd = basecmd % (args.imsize, model, args.last_only, gpus[gpu_idx])
        slots[gpu_idx].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)


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


if args.recursive == 0:
    model_name = args.model.replace("expr/", "").replace("/", "_")

    #generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
    upsample = int(np.log2(args.imsize // 4))
    generator = model.simple.Generator(upsample=upsample)
    missed = generator.load_state_dict(torch.load(args.model), strict=False)
    print(missed)
    generator.to(device)

    print(model_name)

    evaluator = evaluate.LinearityEvaluator(generator, external_model,
    train_iter=args.train_iter,
    test_size=args.test_size,
    latent_dim=128)
    evaluator(generator, model_name)
    exit(0)


# endlessly evaluate if there is new model
st = args.start
while args.recursive == 2:
    model_files = glob.glob(args.model + "/*.model")
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
    if st >= len(model_files):
        break
    model_file = model_files[st]
    model_name = model_file.replace("expr/", "").replace("/", "_")
    print(model_name)
    #generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
    upsample = int(np.log2(args.imsize // 4))
    generator = model.simple.Generator(upsample=upsample)
    missed = generator.load_state_dict(torch.load(model_file), strict=False)
    print(missed)
    generator.to(device)

    evaluator = evaluate.LinearityEvaluator(generator, external_model,
        last_only=args.last_only,
        train_iter=args.train_iter,
        test_size=args.test_size,
        latent_dim=128)
    evaluator(generator, model_name)
    st += 1