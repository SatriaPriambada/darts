import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np

from matplotlib.patches import Patch


def device_color_code(device):
    if device == "cpu-i7-4578U":
        return "red"
    elif device == "gpu-rtx2080":
        return "green"
    elif device == "gpu-v100":
        return "blue"
    elif device == "cpu-x6132":
        return "brown"


def draw_acc_subplot(df, subplot, color, line=False):
    y = df["acc"]
    x = df["mean_lat"]
    e = df["std_dev_lat"]
    if line:
        line = ""
        color = "black"
    else:
        line = ""
    subplot.errorbar(
        x,
        y,
        xerr=e,
        fmt=".",
        mfc=color,
        mec=color,
        linestyle=line,
        ecolor=[color],
        label=args.device,
        color=color,
    )
    subplot.set_title("{} Params vs Latency Behaviour".format(args.device))
    subplot.set_ylabel("acc(%)", fontsize=13)
    subplot.set_xlabel("latency(ms)", fontsize=13)

    locs = subplot.get_xticks()
    x_ticklabels = [str(i) for i in locs]
    subplot.set_xticklabels(x_ticklabels)


def draw_errorbar_graph(file_name, device, path):
    df = pd.read_csv(file_name)
    df_mcts = pd.read_csv(args.filename_mcts)
    print(df)
    print(df_mcts)
    print("start visualizing {} device {}".format(args.filename_mcts, args.device))
    print("result image can be seen in path ./{}".format(args.path))

    plt.figure(figsize=(8, 5))
    plt.clf()
    labels = []
    figure, axes = plt.subplots(nrows=1, ncols=1)
    color = device_color_code(device)
    draw_acc_subplot(df, axes, color, False)
    draw_acc_subplot(df_mcts, axes, color, True)

    figure.tight_layout(pad=0.3)
    name_without_csv = args.filename_mcts.split(".")[0]
    # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    save_name = path + "/{}_cloud".format(name_without_csv)
    plt.savefig(save_name + ".pdf", ext="pdf", bbox_inches="tight")
    plt.savefig(save_name + ".png", ext="png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fc",
        "--filename_cloud",
        type=str,
        default="mcts_generated/arch_profile_arch_op_gap_cifar100_with_acc.csv",
        help="start system with test config",
    )
    parser.add_argument(
        "-fmcts",
        "--filename_mcts",
        type=str,
        default="mcts_generated/arch_profile_mcts_v7_t8_generated_cifar_macro_mcts_v7_sim_100_mcts_architecture_cpu_layers_cifar100_mcts_with_acc.csv",
        help="start system with test config",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu-rtx2080",
        help="device used for profile",
    )
    parser.add_argument(
        "-p", "--path", type=str, default="img", help="path to pdf image results"
    )
    parser.add_argument("--latency_only", action="store_true", default=False)
    args = parser.parse_args()

    draw_errorbar_graph(args.filename_cloud, args.device, args.path)
