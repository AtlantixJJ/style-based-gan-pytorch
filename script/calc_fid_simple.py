import sys, argparse, torch
import model, fid

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="../datasets/CelebAMask-HQ/CelebA-HQ-img-256")
parser.add_argument("--imsize", default=128, type=int)
parser.add_argument("--model", default="checkpoint/fixseg.model")
parser.add_argument("--recursive", default=1, type=int)
args = parser.parse_args()


#generator = model.tfseg.StyledGenerator(semantic="mul-16-none_sl0")
upsample = int(np.log2(cfg.imsize // 4))
generator = model.simple.Generator(upsample=upsample)
generator.load_state_dict(torch.load(args.model), strict=False)
generator.cuda()

evaluator = fid.FIDEvaluator(args.dataset)
fid_value = evaluator(fid.GeneratorIterator(generator, batch_size=1, tot_num=30000, dim=512))
print('FID: ', fid_value)

sg = model.simple.Generator(upsample=upsample)
if len(cfg.gen_load_path) > 0:
    state_dict = torch.load(cfg.gen_load_path, map_location='cpu')
    sg.load_state_dict(state_dict)
    del state_dict