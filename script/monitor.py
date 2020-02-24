"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from os.path import join as osj
from sklearn.metrics.pairwise import cosine_similarity
import evaluate, utils, config, dataset
from lib.face_parsing import unet
import model, segmenter
from lib.netdissect.segviz import segment_visualization_single
from model.semantic_extractor import get_semantic_extractor

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="log,seg,fastagreement", help="")
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--recursive", default="0")
args = parser.parse_args()
print(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.recursive == "1":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files = [f for f in files if os.path.isdir(f"{args.model}/{f}")]
    files.sort()
    gpus = args.gpu.split(",")
    slots = [[] for _ in gpus]
    for i, f in enumerate(files):
        basecmd = "python script/monitor.py --task %s --model %s --gpu %s"
        basecmd = basecmd % (args.task, osj(args.model, f), gpus[i % len(gpus)])
        slots[i % len(gpus)].append(basecmd)
    
    for s in slots:
        cmd = " && ".join(s) + " &"
        print(cmd)
        os.system(cmd)
    exit(0)

# for models store in expr, write result to results; for others, store in same dir
savepath = args.model.replace("expr/", "results/")

device = 'cuda' if int(args.gpu) > -1 else 'cpu'
idmap = utils.CelebAIDMap()
cfg = 0
batch_size = 0
model_path = ""
task = ""
colorizer = utils.Colorize(16) #label to rgb
if "simpleseg" in args.model:
    from model.simple import Generator
    generator = Generator(upsample=4, out_act="none", out_dim=16).to(device)
    model_path = "checkpoint/faceparse_unet_128.pth"
    batch_size = 64
    latent_size = 128
elif "simple" in args.model:
    from model.simple import Generator
    generator = Generator(upsample=4).to(device)
    model_path = "checkpoint/faceparse_unet_128.pth"
    batch_size = 64
    latent_size = 128
elif "stylegan" in args.model:
    if "bedroom" in args.model:
        task = "bedroom"
        colorizer = lambda x: segment_visualization_single(x, 256)
        model_path = "checkpoint/bedroom_lsun_256x256_stylegan.pth"
    elif "celebahq" in args.model:
        task = "celebahq"
        model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
    generator = model.load_stylegan(model_path).to(device)
    model_path = "checkpoint/faceparse_unet_512.pth"
    batch_size = 2
    latent_size = 512
elif "proggan" in args.model:
    if "bedroom" in args.model:
        colorizer = lambda x: segment_visualization_single(x, 256)
        task = "bedroom"
        model_path = "checkpoint/bedroom_lsun_256x256_proggan.pth"
    generator = model.load_proggan(model_path).to(device)
    model_path = "checkpoint/faceparse_unet_512.pth"
    batch_size = 2
    latent_size = 512


def get_extractor_name(model_path):
    keywords = ["nonlinear", "linear", "generative", "cascade"]
    for k in keywords:
        if k in model_path:
            return k


external_model = segmenter.get_segmenter(task, model_path)
n_class = len(external_model.get_label_and_category_names()[1]) + 1
utils.set_seed(65537)
latent = torch.randn(1, latent_size).to(device)
noise = False
op = getattr(generator, "generator_noise", None)
if callable(op):
    noise = op()

with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]

model_files = glob.glob(args.model + "/*.model")
model_files = [m for m in model_files if "disc" not in m]
model_files.sort()
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

if "fast" in args.task:
    LEN = 50
else:
    LEN = 1000


if "log" in args.task:
    logfile = args.model + "/log.txt"
    dic = utils.parse_log(logfile)
    utils.plot_dic(dic, args.model, savepath + "_loss.png")


if "celeba-evaluator" in args.task:
    recordfile = args.model + "/training_evaluation.npy"
    metric = evaluate.SimpleIoUMetric()
    metric.result = np.load(recordfile, allow_pickle=True)[0]
    metric.aggregate(start=len(metric.result) // 2)
    global_dic = metric.global_result
    class_dic = metric.class_result
    print(metric)
    global_result, class_result = metric.aggregate_process()
    utils.plot_dic(global_result, "", savepath + "_global_process.png")
    utils.plot_dic(class_result, "", savepath + "_class_process.png")


if "layer-conv" in args.task:
    colorizer = utils.Colorize(16) #label to rgb
    model_file = model_files[-1]
    latent = torch.randn(64, latent_size, device=device)[45:46]
    state_dict = torch.load(model_file, map_location='cpu')
    missed = generator.load_state_dict(state_dict, strict=False)
    if len(missed.missing_keys) > 1:
        print(missed)
        exit()
    generator.eval()
    generator = generator.to(device)

    image = generator(latent, seg=False)
    image = image.clamp(-1, 1)
    unet_seg = faceparser(F.interpolate(image, size=512, mode="bilinear"))
    unet_label = utils.idmap(unet_seg.argmax(1))
    unet_label_viz = colorizer(unet_label).float() / 255.
    image = (1 + image[0]) / 2
    segs = generator.extract_segmentation(generator.stage)
    LEN = (len(segs) + 1) // 2
    layer_segs = segs[:LEN]
    sum_segs = segs[0:1] + segs[LEN:]
    final_label_viz = colorizer(segs[-1].argmax(1)).float() / 255.
    images = [image, unet_label_viz, final_label_viz]

    prev_seg = 0
    for i, (s, ss) in enumerate(zip(layer_segs, sum_segs)):
        #layer_label = F.interpolate(s, size=image.shape[2], mode="bilinear").argmax(1)[0]
        layer_label = s.argmax(1)[0]
        layer_label_viz = colorizer(layer_label).float() / 255.
        #sum_layers = [F.interpolate(x, size=s.shape[2], mode="bilinear") for x in segs[:i]]
        #sum_layers = sum(sum_layers) + s
        #sum_layers = F.interpolate(sum_layers, size=image.shape[2], mode="bilinear")

        if prev_seg is 0:
            prev_seg = ss
        
        prev_label = F.interpolate(prev_seg, size=s.shape[2], mode="bilinear").argmax(1)[0]

        sum_label = ss.argmax(1)[0]
        sum_label_viz = colorizer(sum_label).float() / 255.
        diff_label_viz = sum_label_viz.clone()
        for i in range(3):
            diff_label_viz[i, :, :][sum_label == prev_label] = 1
        if layer_label_viz.shape[2] < 256:
            layer_label_viz = F.interpolate(
                layer_label_viz.unsqueeze(0), size=256, mode="nearest")[0]
            sum_label_viz = F.interpolate(
                sum_label_viz.unsqueeze(0), size=256, mode="nearest")[0]
            diff_label_viz = F.interpolate(
                diff_label_viz.unsqueeze(0), size=256, mode="nearest")[0]
        images.extend([layer_label_viz, sum_label_viz, diff_label_viz])
        prev_seg = ss
    images = [F.interpolate(img.unsqueeze(0), size=256, mode="bilinear") for img in images]
    images = torch.cat(images)
    print(f"=> Image write to {savepath}_layer-conv.png")
    vutils.save_image(images, f"{savepath}_layer-conv.png", nrow=3)


if "celeba-trace" in args.task:
    trace_path = f"{args.model}/trace_weight.npy"
    trace = np.load(trace_path) # (N, 16, D)
    segments = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    dims = np.cumsum(segments[::-1])
    ind = dims.searchsorted(trace.shape[2])
    ind = len(segments) - ind - 1
    segments = np.cumsum(segments[ind:])
    assert trace.shape[2] == segments[-1]
    weight = trace[-1]

    # variance
    """
    weight_var = trace.std(0)
    fig = plt.figure(figsize=(16, 16))
    maximum, minimum = weight_var.max(), weight_var.min()
    for j in range(16):
        ax = plt.subplot(4, 4, j + 1)
        ax.plot(weight_var[j])
        for x in segments:
            ax.axvline(x=x, c="red", ls="-")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylim([minimum, maximum])
    fig.savefig(f"{savepath}_trace_var.png", bbox_inches='tight')
    plt.close()
    """

    # weight vector
    fig = plt.figure(figsize=(16, 16))
    maximum, minimum = weight.max(), weight.min()
    for j in range(16):
        ax = plt.subplot(4, 4, j + 1)
        ax.plot(weight[j])
        for x in segments:
            ax.axvline(x=x, c="red", ls="-")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylim([minimum, maximum])
    fig.savefig(f"{savepath}_trace_weight.png", bbox_inches='tight')
    plt.close()

    # orthogonal status
    dic = {}
    for i in range(len(segments) - 1):
        prev_dim, cur_dim = segments[i:i+2]
        w = weight[:, prev_dim:cur_dim]
        cosim = cosine_similarity(w, w)
        dic[f"stage{i + 1}"] = cosim
        print(cosim.shape)
    utils.plot_heatmap(dic, "cosim", f"{savepath}_cosim.png")

    # one positive status
    dic = {}
    for i in range(len(segments) - 1):
        prev_dim, cur_dim = segments[i:i+2]
        w = weight[:, prev_dim:cur_dim]
        w[w < 0] = 0 # relu
        op_ratio = w.max(0) / w.sum(0)
        print(op_ratio.shape)
        dic[f"stage{i + 1}"] = op_ratio
    utils.plot_dic(dic, "one positive ratio", f"{savepath}_opr.png")


if "contribution" in args.task:
    latent = torch.randn(1, latent_size, device=device)
    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        state_dict = torch.load(model_file, map_location='cpu')
        missed = generator.load_state_dict(state_dict, strict=False)
        if len(missed.missing_keys) > 1:
            print(missed)
            exit()
        generator.eval()
        w = generator.semantic_branch.weight[:, :, 0, 0]
        segments = generator.semantic_branch.segments
        contrib = torch.zeros(LEN, w.shape[0], w.shape[1])

        for i in tqdm.tqdm(range(LEN)):
            latent.normal_()
            _, gen_seg_logit = generator(latent, detach=True)
            onehot_label = utils.onehot_logit(gen_seg_logit)
            for j in range(onehot_label.shape[1]):
                for s, (bg,ed) in zip(generator.stage, segments):
                    if onehot_label[:, j:j+1].sum() < 1:
                        continue
                    # (1, 1, h, w)
                    weight = utils.adaptive_sumpooling(onehot_label[:, j:j+1], s.shape[3])
                    weight = weight[0, 0].long()
                    mask = weight > 0
                    weight = weight.float() / weight.sum()
                    # (C, N) * (1, N)
                    data_mean = (s[0, :, mask] * weight[mask].unsqueeze(0)).sum(1)
                    contrib[i, j, bg:ed] = (data_mean * w[j, bg:ed]).detach()
        contrib /= 1e-5 + contrib.sum(2, keepdim=True)

    np.save(savepath + "_contrib", contrib.detach().cpu().numpy())


if "agreement" in args.task:
    model_file = model_files[-1]
    latent = torch.randn(batch_size, latent_size, device=device)
    if noise:
        generator.set_noise(noise)
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    sep_model.eval()
    print("=> Load from %s" % model_file)
    state_dict = torch.load(model_file, map_location='cpu')
    missed = sep_model.load_state_dict(state_dict)

    evaluator = evaluate.MaskCelebAEval()
    for i in tqdm.tqdm(range(30 * LEN // batch_size)):
        with torch.no_grad():
            gen, stage = generator.get_stage(latent)
            est_label = sep_model.predict(stage)
            label = external_model.segment_batch(gen.clamp(-1, 1))
        label = utils.torch2numpy(label)

        for j in range(batch_size):
            score = evaluator.calc_single(est_label[j], label[j])
        latent.normal_()

    evaluator.aggregate()
    clean_dic = evaluator.summarize()
    np.save(savepath + "_agreement", clean_dic)

if "seg" in args.task:
    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        sep_model = get_semantic_extractor(get_extractor_name(model_file))(
            n_class=n_class,
            dims=dims).to(device)
        sep_model.eval()
        state_dict = torch.load(model_file, map_location='cpu')
        missed = sep_model.load_state_dict(state_dict)
        sep_model.to(device).eval()

        gen, stage = generator.get_stage(latent, detach=True)
        gen = gen.clamp(-1, 1)
        segs = sep_model(stage)
        segs = [s[0].argmax(0) for s in segs]
        label = external_model.segment_batch(gen)

        segs += [label]

        segs = [colorizer(s).float() / 255. for s in segs]

        res = segs + [(gen[0] + 1) / 2]
        res = [F.interpolate(m.unsqueeze(0), 256).cpu()[0] for m in res]
        fpath = savepath + '{}_segmentation.png'.format(i)
        print("=> Write image to %s" % fpath)
        vutils.save_image(res, fpath, nrow=4)