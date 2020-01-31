"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as osj
from torchvision import utils as vutils
import utils, config
from lib.face_parsing import unet

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="agreement", help="")
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--zero", type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.model == "expr":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files.sort()
    for f in files:
        basecmd = "python script/monitor.py --task %s --model %s --gpu %s --zero %d"
        basecmd = basecmd % (args.task, osj(args.model, f), args.gpu, args.zero)
        os.system(basecmd)
    exit(0)

savepath = args.model.replace("expr/", "results/")

device = 'cuda' if int(args.gpu) > -1 else 'cpu'
torch.manual_seed(1116)

cfg = 0
batch_size = 0
if "simpleseg" in args.model:
    from model.simple import Generator
    generator = Generator(upsample=4, out_act="none", out_dim=16).to(device)
    imsize = 128
    batch_size = 64
    latent_size = 128
elif "simple" in args.model:
    from model.simple import Generator
    generator = Generator(upsample=4).to(device)
    imsize = 128
    batch_size = 64
    latent_size = 128
else:
    cfg = config.config_from_name(args.model)
    print(cfg)
    from model.tfseg import StyledGenerator
    generator = StyledGenerator(**cfg).to(device)
    imsize = 512
    batch_size = 2
    latent_size = 512
faceparser_path = f"checkpoint/faceparse_unet_{imsize}.pth"

latent = torch.randn(1, latent_size).to(device)
latent.requires_grad = True
noise = []
for i in range(18):
    size = 4 * 2 ** (i // 2)
    noise.append(torch.randn(1, 1, size, size, device=device))


if args.zero:
    print("=> Use zero as noise")
    noise = [0] * 18
    for k in range(18):
        size = 4 * 2 ** (k // 2)
        noise[k] = torch.zeros(1, 1, size, size).to(device)
    generator.set_noise(noise)

model_files = glob.glob(args.model + "/*.model")
model_files = [m for m in model_files if "disc" not in m]
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

state_dict = torch.load(faceparser_path, map_location='cpu')
faceparser = unet.unet()
faceparser.load_state_dict(state_dict)
faceparser = faceparser.to(device)
faceparser.eval()
del state_dict

if "log" in args.task:
    logfile = args.model + "/log.txt"
    dic = utils.parse_log(logfile)
    utils.plot_dic(dic, args.model, savepath + "_loss.png")

if "agreement" in args.task:
    latent = torch.randn(batch_size, latent_size, device=device)
    dics = []
    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        state_dict = torch.load(model_file, map_location='cpu')
        missed = generator.load_state_dict(state_dict, strict=False)
        if len(missed.missing_keys) > 1:
            print(missed)
            exit()
        generator.eval()

        evaluator = utils.MaskCelebAEval(map_id=True)
        for i in tqdm.tqdm(range(32 * 1000 // batch_size)):
            gen, gen_seg_logit = generator(latent)
            gen_seg_logit = F.interpolate(gen_seg_logit, imsize, mode="bilinear")
            seg = gen_seg_logit.argmax(1)
            gen = gen.clamp(-1, 1)

            with torch.no_grad():
                gen = F.interpolate(gen, imsize, mode="bilinear")
                label = faceparser(gen).argmax(1)
                label = utils.idmap(label)
            
            seg = utils.torch2numpy(seg)
            label = utils.torch2numpy(label)

            for j in range(batch_size):
                score = evaluator.compute_score(seg[j], label[j])
                evaluator.accumulate(score)
            latent.normal_()
        evaluator.aggregate()
        dics.append(evaluator.summarize())

    new_dic = {k: [d[k] for d in dics] for k in dics[0].keys()}
    np.save(savepath + "_agreement", new_dic)
    utils.format_agreement_result(new_dic)

if "seg" in args.task:
    colorizer = utils.Colorize(16) #label to rgb

    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        state_dict = torch.load(model_file, map_location='cpu')
        missed = generator.load_state_dict(state_dict, strict=False)
        print(missed)
        generator.eval()

        gen = generator(latent, False)
        gen = gen.clamp(-1, 1)
        segs = generator.extract_segmentation(generator.stage)
        segs = [s[0].argmax(0) for s in segs]

        with torch.no_grad():
            gen = F.interpolate(gen, imsize, mode="bilinear")
            label = faceparser(gen)[0].argmax(0)
            label = utils.idmap(label)
        
        segs += [label]

        segs = [colorizer(s).float() / 255. for s in segs]

        res = segs + [(gen[0] + 1) / 2]
        res = [F.interpolate(m.unsqueeze(0), 256).cpu()[0] for m in res]
        fpath = savepath + '{}_segmentation.png'.format(i)
        print("=> Write image to %s" % fpath)
        vutils.save_image(res, fpath, nrow=4)