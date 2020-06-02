"""
Monitor the training, visualize the trained model output.
"""
import sys
sys.path.insert(0, ".")
import argparse, tqdm, glob, os, copy
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as osj
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="log,seg,fast-celeba-agreement,celeba-evaluator", help="")
parser.add_argument("--model", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--recursive", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import evaluate, utils, config, dataset
import model, segmenter
from lib.netdissect.segviz import segviz_torch
from model.semantic_extractor import get_semantic_extractor
from weight_visualization import *

if args.recursive == "1":
    # This is root, run for all the expr directory
    files = os.listdir(args.model)
    files = [f for f in files if os.path.isdir(f"{args.model}/{f}") and os.path.exists(f"{args.model}/{f}/log.txt")]
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
#idmap = utils.CelebAIDMap()
cfg = 0
batch_size = 0
model_path = ""
task = ""
colorizer = utils.Colorize(15) #label to rgb
if "simpleseg" in args.model:
    from model.simple import Generator
    generator = Generator(upsample=4, out_act="none", out_dim=16).to(device)
    model_path = "checkpoint/faceparse_unet_128.pth"
    batch_size = 64
    latent_size = 128
elif "wgan64" in args.model:
    from model.simple import Generator
    generator = Generator(size=64).to(device)
    model_path = "checkpoint/faceparse_unet_128.pth"
    task = "celebahq"
    batch_size = 64
    latent_size = 128
elif "wgan128" in args.model:
    from model.simple import Generator
    generator = Generator(size=128).to(device)
    model_path = "checkpoint/faceparse_unet_128.pth"
    task = "celebahq"
    batch_size = 16
    latent_size = 128
elif "stylegan2" in args.model:
    colorizer = segviz_torch
    if "bedroom" in args.model:
        task = "bedroom"
        model_path = "checkpoint/bedroom_lsun_256x256_stylegan2.pth"
    elif "church" in args.model:
        task = "church"
        model_path = "checkpoint/church_lsun_256x256_stylegan2.pth"
    elif "ffhq" in args.model:
        task = "ffhq"
        colorizer = utils.Colorize(15)
        model_path = "checkpoint/face_ffhq_1024x1024_stylegan2.pth"
    elif "cat" in args.model:
        task = "cat"
        model_path = "checkpoint/cat_lsun_256x256_stylegan2.pth"
    elif "car" in args.model:
        task = "car"
        model_path = "checkpoint/car_lsun_512x384_stylegan2.pth"
    elif "horse" in args.model:
        task = "horse"
        model_path = "checkpoint/horse_lsun_256x256_stylegan2.pth"
    generator = model.load_stylegan2(model_path).to(device)
    model_path = "checkpoint/faceparse_unet_512.pth"
    batch_size = 1
    latent_size = 512
elif "stylegan" in args.model:
    if "bedroom" in args.model:
        task = "bedroom"
        colorizer = segviz_torch
        model_path = "checkpoint/bedroom_lsun_256x256_stylegan.pth"
    elif "celebahq" in args.model:
        task = "celebahq"
        model_path = "checkpoint/face_celebahq_1024x1024_stylegan.pth"
    generator = model.load_stylegan(model_path).to(device)
    model_path = "checkpoint/faceparse_unet_512.pth"
    batch_size = 1
    latent_size = 512
elif "proggan" in args.model:
    if "bedroom" in args.model:
        colorizer = segviz_torch
        task = "bedroom"
        model_path = "checkpoint/bedroom_lsun_256x256_proggan.pth"
    if "church" in args.model:
        colorizer = segviz_torch
        task = "church"
        model_path = "checkpoint/church_lsun_256x256_proggan.pth"
    if "celebahq" in args.model:
        task = "celebahq"
        model_path = "checkpoint/face_celebahq_1024x1024_proggan.pth"
    generator = model.load_proggan(model_path).to(device)
    model_path = "checkpoint/faceparse_unet_512.pth"
    batch_size = 1
    latent_size = 512 

def get_extractor_name(model_path):
    keywords = ["nonlinear", "linear", "spherical", "generative", "projective", "unitnorm", "unit"]
    for k in keywords:
        if k in model_path:
            return k

external_model = segmenter.get_segmenter(task, model_path, device)
labels, cats = external_model.get_label_and_category_names()
category_groups = utils.get_group(labels)
category_groups_label = utils.get_group(labels, False)
n_class = category_groups[-1][1]
print(category_groups)
utils.set_seed(65537)
latent = torch.randn(1, latent_size).to(device)
noise = False
op = getattr(generator, "generate_noise", None)
if callable(op):
    noise = op(device)

with torch.no_grad():
    image, stage = generator.get_stage(latent)
dims = [s.shape[1] for s in stage]
layers = list(range(9))
if "layer" in args.model:
    ind = args.model.rfind("layer") + len("layer")
    s = args.model[ind:].split("_")[0]
    layers = [int(i) for i in s.split(",") if int(i) < len(dims)]
    dims = np.array(dims)[layers].tolist()
print(dims)

model_files = glob.glob(args.model + "/*.model")
if ".model" not in args.model:
    model_files = [m for m in model_files if "disc" not in m]
    model_files.sort()
else:
    model_files = [args.model]
    
if len(model_files) == 0:
    print("!> No model found, exit")
    exit(0)

if "fast" in args.task:
    LEN = 100
else:
    LEN = 400


if "log" in args.task:
    logfile = args.model + "/log.txt"
    dic = utils.parse_log(logfile)
    utils.plot_dic(dic, args.model, savepath + "_loss.png")

if "seg" in args.task:
    for i, model_file in enumerate(model_files):
        print("=> Load from %s" % model_file)
        sep_model = get_semantic_extractor(get_extractor_name(model_file))(
            n_class=n_class,
            category_groups=category_groups,
            dims=dims).to(device)
        sep_model.eval()
        state_dict = torch.load(model_file, map_location='cpu')
        missed = sep_model.load_state_dict(state_dict)
        sep_model.to(device).eval()
        with torch.no_grad():
            gen, stage = generator.get_stage(latent, detach=True)
        stage = [s for i, s in enumerate(stage) if i in layers]
        gen = gen.clamp(-1, 1)
        segs = sep_model(stage)
        if "face" in model_file:
            segs = [s.argmax(1) for s in segs]
        else:
            for s in segs[0]:
                print(s.shape)
            segs = [s.argmax(1) for s in segs[0]]
        label = external_model.segment_batch(gen)

        segs += [label]
        for s in segs: print(s.shape)
        segs = [colorizer(s).float() / 255. for s in segs]
        res = segs + [(gen[0:1] + 1) / 2]

        res = [F.interpolate(m, 256).cpu()[0] for m in res]
        fpath = savepath + '{}_segmentation.png'.format(i)
        print("=> Write image to %s" % fpath)
        vutils.save_image(res, fpath, nrow=4)
        
if "projective" in args.task:
    H, W = image.shape[2:]

    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    orig_weight = torch.load(model_file, map_location=device)
    sep_model.load_state_dict(orig_weight)

    for ind in range(4):
        with torch.no_grad():
            latent.normal_()
            image, stage = generator.get_stage(latent)
            image = (image.clamp(-1, 1) + 1) / 2
            seg, projection = sep_model(stage, True, True)
            pred = seg[0].argmax(1)
            pred_viz = colorizer(pred).float() / 255.
            projection = F.interpolate(projection, 256, mode="bilinear", align_corners=True)
            pred = F.interpolate(seg[0], 256, mode="bilinear", align_corners=True).argmax(1)
            c = (np.array(utils.CELEBA_COLORS)/255.)[pred.view(-1).tolist()]
            x = projection[0, 0].view(-1).cpu()
            y = projection[0, 1].view(-1).cpu()
            plt.scatter(x, y, s=1, c=c)
            plt.tight_layout()
            plt.savefig("test.png")
            plt.close()

            vutils.save_image(utils.catlist([image, pred_viz]), "image.png")


if "celeba-evaluator" in args.task:
    recordfile = args.model + "/training_evaluation.npy"
    metric = evaluate.SimpleIoUMetric(ignore_classes=[0])
    metric.result = np.load(recordfile, allow_pickle=True)[0]
    metric.aggregate(start=len(metric.result) // 2)
    global_dic = metric.global_result
    class_dic = metric.class_result
    print(metric)
    global_result, class_result = metric.aggregate_process()
    utils.plot_dic(global_result, "", savepath + "_global_process.png")
    utils.plot_dic(class_result, "", savepath + "_class_process.png")


if "layer-conv" in args.task:
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
    unet_seg = faceparser(F.interpolate(image, size=512, mode="bilinear", align_corners=True))
    unet_label = unet_seg.argmax(1)
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
        #layer_label = F.interpolate(s, size=image.shape[2], mode="bilinear", align_corners=True).argmax(1)[0]
        layer_label = s.argmax(1)[0]
        layer_label_viz = colorizer(layer_label).float() / 255.
        #sum_layers = [F.interpolate(x, size=s.shape[2], mode="bilinear", align_corners=True) for x in segs[:i]]
        #sum_layers = sum(sum_layers) + s
        #sum_layers = F.interpolate(sum_layers, size=image.shape[2], mode="bilinear", align_corners=True)

        if prev_seg is 0:
            prev_seg = ss
        
        prev_label = F.interpolate(prev_seg, size=s.shape[2], mode="bilinear", align_corners=True).argmax(1)[0]

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
    images = [F.interpolate(img.unsqueeze(0), size=256, mode="bilinear", align_corners=True) for img in images]
    images = torch.cat(images)
    print(f"=> Image write to {savepath}_layer-conv.png")
    vutils.save_image(images, f"{savepath}_layer-conv.png", nrow=3)


def sample_cosine(data, pred, n_class=16):
    mean_table = np.zeros((n_class, n_class))
    std_table = np.zeros((n_class, n_class))
    size_table = np.zeros((n_class, n_class))
    #cosim_table = [[[] for _ in range(n_class)] for _ in range(n_class)]
    class_indice = [-1] * n_class
    class_arr = [-1] * n_class
    class_length = [-1] * n_class
    N = 100000
    for i in range(1, n_class): # no background
        mask = pred == i
        length = mask.sum()
        if length == 0:
            pass # no this class
        elif length < N:
            class_arr[i] = data[mask]
        else:
            idx = np.where(mask)
            indice = np.random.choice(
                np.arange(0, len(idx[0])),
                N)
            class_arr[i] = np.stack([data[idx[0][i], idx[1][i]]
                for i in indice])
        class_length[i] = length
    
    print("=> Class number:")
    for i in range(n_class):
        print("=> %s: %s" % (utils.CELEBA_CATEGORY[i], str(class_length[i])))

    for i in range(1, n_class):
        if class_length[i] < 1:
            continue
        class_arr[i] /= np.linalg.norm(class_arr[i], 2, 1, True)
    print("=> Normalization done")

    del data

    for i in range(1, n_class):
        if class_length[i] < 1:
            continue
        for j in range(i, n_class):
            if class_length[j] < 1:
                continue
            cosim = np.matmul(class_arr[i], class_arr[j].transpose())
            mean_table[j, i] = mean_table[i, j] = cosim.mean()
            std_table[j, i] = std_table[i, j] = cosim.std()
            size_table[i, j] = size_table[j, i] = np.prod(cosim.shape)
            #cosim_table[i][j] = cosim.reshape(-1).copy()
            del cosim
    return mean_table, std_table, size_table


if "cosim" in args.task:
    H, W = image.shape[2:]

    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    orig_weight = torch.load(model_file, map_location=device)
    sep_model.load_state_dict(orig_weight)
    for ind in range(32):
        with torch.no_grad():
            latent.normal_()
            image, stage = generator.get_stage(latent)
            seg = sep_model(stage, True)[0]
            pred = seg.argmax(1)
            pred_viz = colorizer(pred).float() / 255.
            pred = utils.torch2numpy(pred[0])
            image = (1 + image.clamp(-1, 1)) / 2
            vutils.save_image(
                utils.catlist([image, pred_viz]),
                f"{savepath}_{ind}_imagelabel.png")
            data = 0
            if "cosim-feature" in args.task:
                print("=> Interpolating large feature")
                data = torch.cat([F.interpolate(s.cpu(), size=H, mode="bilinear", align_corners=True)[0] for s in stage]).permute(1, 2, 0)
                data = utils.torch2numpy(data)
                np.save(f"feats_{ind}.npy", data)
            elif "cosim-calc" in args.task:
                data = np.load(f"feats_{ind}.npy", allow_pickle=True)
            mean_table, std_table, size_table = sample_cosine(data, pred, n_class)
            np.save(f"{savepath}_{ind}_cosim.npy", [mean_table, std_table, size_table])


if "score-second" in args.task:
    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    orig_weight = torch.load(model_file, map_location=device)
    sep_model.load_state_dict(orig_weight)
    images = []
    for i in range(8):
        with torch.no_grad():
            image, stage = generator.get_stage(latent)
            seg1 = sep_model(stage, True)[0]
            max_pred = seg1.argmax(1)
            seg2 = seg1.detach().clone()
            seg2.scatter_(1,
                max_pred.unsqueeze(1),
                torch.zeros_like(seg2).fill_(-100))
            assert seg2[0, max_pred[0, 0, 0], 0, 0] == -100

            sec_pred = seg2.argmax(1)
            max_value = torch.gather(seg1, 1, max_pred.unsqueeze(1))
            sec_value = torch.gather(seg1, 1, sec_pred.unsqueeze(1))
            score_diff = max_value - sec_value
            
            mini, maxi = score_diff.min(), score_diff.max()
            score_diff = (score_diff - mini) / (maxi - mini)
            score_diff_viz = utils.heatmap_torch(score_diff.cpu())

            latent.normal_()

            image = (image[0].clamp(-1, 1).cpu() + 1) / 2
            pred_viz = colorizer(max_pred.cpu()).float() / 255.
            images.extend([image, pred_viz, score_diff_viz[0]])
    images = [F.interpolate(img.unsqueeze(0), size=256) for img in images]
    vutils.save_image(torch.cat(images), f"{savepath}_score_second.png", nrow=6)


if "score-first" in args.task:
    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=16,
        dims=dims).to(device)
    orig_weight = torch.load(model_file, map_location=device)
    sep_model.load_state_dict(orig_weight)
    images = []
    for i in range(8):
        with torch.no_grad():
            image, stage = generator.get_stage(latent)
            seg1 = sep_model(stage, True)[0]
            max_pred = seg1.argmax(1)
            max_value = torch.gather(seg1, 1, max_pred.unsqueeze(1))

            # filtering
            #std = max_value.std() * 2
            #mean = max_value.mean()
            #max_value[max_value > mean + std] = mean + std

            mini, maxi = max_value.min(), max_value.max()
            max_value = (max_value - mini) / (maxi - mini)

            max_value_viz = utils.heatmap_torch(max_value.cpu())

            latent.normal_()

            image = (image[0].clamp(-1, 1).cpu() + 1) / 2
            pred_viz = colorizer(max_pred.cpu()).float() / 255.
            images.extend([image, pred_viz, max_value_viz[0]])
    images = [F.interpolate(img.unsqueeze(0), size=256) for img in images]
    vutils.save_image(torch.cat(images), f"{savepath}_score_first.png", nrow=6)


if "surgery" in args.task:
    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    orig_weight = torch.load(model_file, map_location=device)
    for st in [[4], [5], [6], [7], [8]]:
        new_weight = copy.deepcopy(orig_weight)
        early_layer_surgery(new_weight, st)
        subfix = f"_e{st}_n"
        sep_model.load_state_dict(new_weight)
        if "spherical" not in model_file:
            mini, maxi = find_min_max_weight(
                sep_model.semantic_extractor)
            plot_weight_layerwise(
                sep_model.semantic_extractor,
                mini, maxi, subfix)
        latent = torch.randn(1, latent_size, device=device)
        images = []
        for i in range(8):
            with torch.no_grad():
                image, stage = generator.get_stage(latent)
                sep_model.load_state_dict(orig_weight)
                old_pred = sep_model.predict(stage)
                sep_model.load_state_dict(new_weight)
                new_pred = sep_model.predict(stage)

                latent.normal_()

                print(evaluate.iou(old_pred, new_pred))   

                image = (image[0].clamp(-1, 1).cpu() + 1) / 2
                old_pred_viz = colorizer(torch.from_numpy(old_pred)).float() / 255.
                new_pred_viz = colorizer(torch.from_numpy(new_pred)).float() / 255.
                diff_viz = torch.ones_like(new_pred_viz)
                mask = old_pred[0] != new_pred[0]
                diff_viz[:, mask] = new_pred_viz[:, mask]
                images.extend([image, old_pred_viz, diff_viz])
        images = [F.interpolate(img.unsqueeze(0), size=256) for img in images]
        vutils.save_image(torch.cat(images), f"{savepath}_surgery{subfix}.png", nrow=6)


if "weight" in args.task and "nonlinear" not in model_files[-1] and "generative" not in model_files[-1]:
    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims)
    sep_model.load_state_dict(torch.load(model_file))
    
    print("=> Load from %s" % model_file)

    # weight vector
    minimum = maximum = ws = 0
    if "spherical" in model_file:
        ws = sep_model.weight[:, :, 0, 0]
        minimum, maximum = ws.min().detach(), ws.max().detach()
    elif ("unit_extractor" in model_file) or ("unitnorm" in model_file):
        ws = sep_model.weight[:, :, 0, 0]
        ws = F.normalize(ws, 2, 1)
        minimum, maximum = ws.min().detach(), ws.max().detach()
    else:
        minimum, maximum = find_min_max_weight(
            sep_model.semantic_extractor)
        ws = concat_weight(sep_model.semantic_extractor)
    ws = ws.detach()

    fig = plt.figure(figsize=(16, 12))
    for j in range(ws.shape[0]):
        ax = plt.subplot(4, 4, j + 1)
        x = list(range(len(ws[j])))
        ax.scatter(x, ws[j], s=2)
        ax.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{savepath}_weight_class.png", bbox_inches='tight')
    plt.close()
    
    if "spherical" in model_file or "_unit_extractor" in model_file or "unitnorm" in model_file:
        plot_weight_concat(
            ws.detach().numpy(),
            maximum, minimum)
    else:
        plot_weight_layerwise(
            sep_model.semantic_extractor,
            maximum, minimum,
            f"{savepath}_weight_layer")
    s = model_file[model_file.rfind("/")+1:]
    threshold = s + "/threshold.txt"
    if os.path.exists(threshold):
        threshold = open(threshold, "r").read()

    """
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
    """


if "visualize" in args.task:
    H, W = image.shape[2:]
    size = H // 2
    model_file = model_files[-1]
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims).to(device)
    sep_model.eval()
    state_dict = torch.load(model_file, map_location='cpu')
    missed = sep_model.load_state_dict(state_dict)
    sep_model.to(device).eval()
    w = 0
    if "spherical" in model_file:
        w = sep_model.weight[:, :, 0, 0]
    else:
        w = concat_weight(sep_model.semantic_extractor)
    xs, ys = weight_project_direction(w)


    for ind in range(4):
        with torch.no_grad():
            print("=> Preparing feature")
            latent.normal_()
            image, stage = generator.get_stage(latent)
            seg = sep_model(stage, True)[0]
            seg = F.interpolate(seg, size=size, mode="bilinear", align_corners=True)
            pred = seg.argmax(1)
            pred_viz = colorizer(pred).float() / 255.
            data = torch.cat([F.interpolate(s, size=size, mode="bilinear", align_corners=True)[0].cpu() for s in stage]).permute(1, 2, 0)
            c = (np.array(utils.CELEBA_COLORS)/255.)[pred.view(-1).tolist()]
            data = data.view(-1, data.shape[2])

            print("=> Random single class projection")
            fig = plt.figure(figsize=(40, 36))
            vx, vy = projection_direction(data.shape[1])
            for i in range(n_class):
                x, y = get_projection(
                    data[pred.view(-1) == i],
                    vx, vy)
                cs = (np.array(utils.CELEBA_COLORS)/255.)[i]
                ax = plt.subplot(4, 4, 1 + i)
                ax.scatter(x, y, s=1, c=cs.reshape(1, 3))
            plt.savefig(f"random_single_projection_{ind}.png", box_inches="tight")
            plt.close()

            print("=> Random projection")
            fig = plt.figure(figsize=(40, 36))
            x, y = get_projection(data, vx, vy)
            plt.scatter(x, y, s=1, c=c)
            plt.savefig(f"random_projection_{ind}.png", box_inches="tight")
            plt.close()

            print("=> Weight single class projection")
            fig = plt.figure(figsize=(40, 36))
            for i in range(n_class):
                x, y = get_projection(
                    data[pred.view(-1) == i],
                    xs[i], ys[i])
                cs = (np.array(utils.CELEBA_COLORS)/255.)[i]
                ax = plt.subplot(4, 4, 1 + i)
                ax.scatter(x, y, s=1, c=cs.reshape(1, 3))
            plt.savefig(f"weight_single_projection_{ind}.png", box_inches="tight")
            plt.close()

            print("=> Weight projection")
            fig = plt.figure(figsize=(40, 36))
            for i in range(n_class):
                x, y = get_projection(data, xs[i], ys[i])
                ax = plt.subplot(4, 4, 1 + i)
                ax.scatter(x, y, c=c, s=1)
            plt.savefig(f"weight_projection_{ind}.png", box_inches="tight")
            plt.close()


    """
    latent.uniform_(); latent.normal_()
    datafiles = glob.glob("../datasets/StyleGANFeats/*.npy")
    datafiles.sort()
    for f in datafiles:
        data = np.load(f, allow_pickle=True)[()].astype("float32")
        x, y = random_projection(data)
        x = x.reshape(-1)
        y = y.reshape(-1)
        plt.scatter(x, y, s=1)
        plt.savefig("test.png")
        plt.close()
    """


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


if "lsun-agreement" in args.task:
    model_file = model_files[-1]
    latent = torch.randn(batch_size, latent_size, device=device)
    if noise:
        generator.set_noise(noise)
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        category_groups=category_groups,
        dims=dims).to(device)
    sep_model.eval()
    print("=> Load from %s" % model_file)
    state_dict = torch.load(model_file, map_location='cpu')
    missed = sep_model.load_state_dict(state_dict)
    metrics = [evaluate.DetectionMetric(n_class=cg[1]-cg[0])
        for i, cg in enumerate(category_groups)]
    for ind in tqdm.tqdm(range(30 * LEN)):
        with torch.no_grad():
            gen, stage = generator.get_stage(latent)
            gen = gen.clamp(-1, 1)
            label = external_model.segment_batch(gen)

        multi_segs = sep_model(stage)
        if len(category_groups_label) == 1:
            multi_segs = [multi_segs]
            label = label.unsqueeze(1)
            
        label = utils.torch2numpy(label)
        for i, seg in enumerate(multi_segs):
            if label[:, i].max() <= 0:
                continue
            cg = category_groups_label[i]
            l = label[:, i, :, :] - cg[0]
            l[l<0] = 0

            # collect training statistic
            est_label = seg[-1].argmax(1)
            for j in range(est_label.shape[0]):
                metrics[i](
                    utils.torch2numpy(est_label[j]),
                    utils.torch2numpy(l[j]))
    # show metric
    for i in range(len(category_groups)):
        metrics[i].aggregate()
        print(metrics[i])
    np.save(
        f"{savepath}_agreement.npy",
        [metrics[i].result for i in range(len(metrics))])


if "celeba-agreement" in args.task:
    model_file = model_files[-1]
    batch_size = 1
    latent = torch.randn(batch_size, latent_size, device=device)
    if noise:
        generator.set_noise(noise)
    sep_model = get_semantic_extractor(get_extractor_name(model_file))(
        n_class=n_class,
        dims=dims,
        use_bias="bias1" in model_file)
    sep_model.to(device).eval()
    print("=> Load from %s" % model_file)
    is_resize = "spherical" not in model_file
    state_dict = torch.load(model_file, map_location='cpu')
    missed = sep_model.load_state_dict(state_dict)
    evaluator = evaluate.MaskCelebAEval()
    for i in tqdm.tqdm(range(30 * LEN)):
        with torch.no_grad():
            gen, stage = generator.get_stage(latent)
            stage = [s for i, s in enumerate(stage) if i in layers]
            gen = gen.clamp(-1, 1)
            est_label = sep_model.predict(stage)
            label = external_model.segment_batch(gen,
                resize=is_resize)
        label = F.interpolate(label.unsqueeze(1).float(),
            size=est_label.shape[2],
            mode="nearest").long()[:, 0, :, :]
        label = utils.torch2numpy(label)

        for j in range(batch_size):
            score = evaluator.calc_single(est_label[j], label[j])
        latent.normal_()

    evaluator.aggregate()
    clean_dic = evaluator.summarize()
    np.save(savepath + "_agreement", clean_dic)


if "gan" in args.task:
    latent = torch.randn(18, latent_size, device=device)
    for idx,model_file in enumerate(model_files):
        print("=> model file %s" % model_file)
        missed = generator.load_state_dict(torch.load(model_file))
        print(missed)
        gen = generator(latent).clamp(-1, 1)
        label = external_model.segment_batch(gen)
        res = []
        gen = (gen.detach().cpu() + 1) / 2
        for i in range(latent.shape[0]):
            label_viz = colorizer(label[i]).unsqueeze(0).float() / 255.
            res.extend([gen[i:i+1], label_viz])
        vutils.save_image(torch.cat(res), f"{savepath}_{idx}_gan.png", nrow=6)
