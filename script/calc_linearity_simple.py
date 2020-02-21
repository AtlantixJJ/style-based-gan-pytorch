"""
python script/calc_linearity_simple.py --imsize 64 --data-dir datasets/Synthesized --test-dir datasets/Synthesized_test --model expr/celeba_hat_wgan64/gen_iter_001000.model --recursive 0
"""
import sys, argparse, torch, glob, os
sys.path.insert(0, ".")
import numpy as np
import model, fid, utils, evaluate, dataset
from lib.face_parsing.unet import unet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", default="datasets/Synthesized")
parser.add_argument(
    "--test-dir", default="datasets/Synthesized_test")
parser.add_argument(
    "--imsize", default=128, type=int)
parser.add_argument(
    "--model", default="checkpoint/fixseg_conv-16-1.model")
parser.add_argument(
    "--recursive", default=1, type=int)
parser.add_argument(
    "--gpu", default="0")
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
        basecmd = "python script/calc_linearity_simple.py --dataset %s --imsize %d --model %s --gpu %s --recursive 0"
        basecmd = basecmd % (args.dataset, args.imsize, model, gpus[i % len(gpus)])
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

"""
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
"""

train_ds = dataset.LatentSegmentationDataset(
    latent_dir=args.data_dir + "/latent",
    noise_dir=args.data_dir + "/noise",
    image_dir=None,
    seg_dir=args.data_dir + "/label")
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False)

test_ds = dataset.LatentSegmentationDataset(
    latent_dir=args.test_dir + "/latent",
    noise_dir=args.test_dir + "/noise",
    image_dir=None,
    seg_dir=args.test_dir + "/label")
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)


evaluator = evaluate.LinearityEvaluator(train_dl, test_dl)
evaluator(generator, model_name)