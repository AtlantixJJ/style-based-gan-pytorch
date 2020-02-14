import sys, argparse, torch, glob, os
import numpy as np
import model, fid

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="../datasets/CelebAMask-HQ/CelebA-HQ-img-128")
parser.add_argument("--imsize", default=128, type=int)
parser.add_argument("--model", default="checkpoint/fixseg.model")
parser.add_argument("--recursive", default=1, type=int)
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if args.recursive == "1":
    # This is root, run for all the expr directory
    model_files = glob.glob(args.model + "/*.model")
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, model in enumerate(model_files):
        basecmd = "python script/calc_fid_simple.py --model %s --gpu %s --recursive 0"
        basecmd = basecmd % (model, gpus[i % len(gpus)])
        slots[i % len(gpus)].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)


#generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
upsample = int(np.log2(args.imsize // 4))
generator = model.simple.Generator(upsample=upsample)
generator.load_state_dict(torch.load(args.model), strict=False)
generator.cuda()

evaluator = fid.FIDEvaluator(args.dataset)
fid_value = evaluator(fid.GeneratorIterator(generator, batch_size=64, tot_num=30000, dim=128))
print('FID: ', fid_value)
