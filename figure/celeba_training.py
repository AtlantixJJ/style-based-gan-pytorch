import sys
sys.path.insert(0, ".")
import glob
import evaluate, utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())


def plot_dic(dic, file=None):
    n = len(dic.items())
    fig = plt.figure(figsize=(4 * 7, 3 * 2))
    for i, (k, v) in enumerate(dic.items()):
        ax = fig.add_subplot(2, 7, i + 1)
        ax.plot(v)
        ax.set_title(k)
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()


dir = "record/celebahq1"
dirs = glob.glob(dir)

for data_dir in dirs:
    model_files = glob.glob(f"{data_dir}/*")
    model_files = [f for f in model_files if "." not in f]
    recordfiles = [f"{f}/training_evaluation.npy" for f in model_files]
    for recordfile in recordfiles:
        name = utils.listkey_convert(recordfile,
            ["nonlinear", "linear", "generative", "spherical"])
        metric = evaluate.SimpleIoUMetric(ignore_classes=[0, 13])
        metric.result = np.load(recordfile, allow_pickle=True)[0]
        metric.aggregate(start=len(metric.result) // 2)
        global_dic = metric.global_result
        class_dic = metric.class_result
        global_result, class_result = metric.aggregate_process()
        del class_result["background"]
        del class_result["neck_l"]
        plot_dic(class_result, f"{name}_class_process.pdf")
