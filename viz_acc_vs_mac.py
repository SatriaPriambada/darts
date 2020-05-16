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
        return "black"
    elif device == "gpu-v100":
        return "blue"
    elif device == "cpu-x6132":
        return "brown"


def draw_acc_subplot(df, subplot, color):
    y = df["acc"]
    x = df["macs"]
    e = 0
    if args.line:
        line = "-"
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
        label="ours",
        color=color,
    )

    subplot.errorbar(
        [300000000, 585000000],
        [72, 74.7],
        xerr=0,
        fmt=".",
        mfc="red",
        mec="red",
        ecolor=["red"],
        color="red",
        linestyle=line,
        label="mobilenetv2",
    )

    subplot.errorbar(
        [574000000],
        [73.3],
        xerr=0,
        fmt=".",
        mfc="green",
        mec="green",
        ecolor=["green"],
        color="green",
        label="DARTS",
    )
    subplot.errorbar(
        [350000000],
        [75.1],
        xerr=0,
        fmt=".",
        mfc="blue",
        mec="blue",
        ecolor=["blue"],
        color="blue",
        label="proxylessnas",
    )
    subplot.errorbar(
        [595000000],
        [80.0],
        xerr=0,
        fmt=".",
        mfc="brown",
        mec="brown",
        ecolor=["brown"],
        color="brown",
        label="OFA",
    )

    subplot.errorbar(
        [588000000],
        [74.2],
        xerr=0,
        fmt=".",
        mfc="#FFA454",
        mec="#FFA454",
        ecolor=["#FFA454"],
        label="PNASNet",
        color="#FFA454",
    )

    subplot.errorbar(
        [56000000, 219000000],
        [67.4, 75.2],
        xerr=0,
        fmt=".",
        mfc="#F0A454",
        mec="#F0A454",
        ecolor=["#F0A454"],
        color="#F0A454",
        linestyle=line,
        label="mobilenetv3",
    )

    subplot.set_title("Acc vs MACs Comparison".format(args.device))
    subplot.set_ylabel("acc(%)", fontsize=13)
    subplot.set_xlabel("MAC(M)", fontsize=13)
    subplot.legend(loc="lower right")
    locs = subplot.get_xticks()
    x_ticklabels = [str(i / 1000000) + "M" for i in locs]
    subplot.set_xticklabels(x_ticklabels)
    return subplot


def draw_errorbar_graph(file_name, device, path):
    df = pd.read_csv(file_name)
    print(df)
    print("start visualizing {} device {}".format(args.filename, args.device))
    print("result image can be seen in path ./{}".format(args.path))

    plt.figure(figsize=(8, 5))
    plt.clf()
    labels = ["ours"]
    figure, axes = plt.subplots(nrows=1, ncols=1)
    color = device_color_code(device)
    axes = draw_acc_subplot(df, axes, color)

    figure.tight_layout(pad=0.3)

    # lgd = plt.legend(
    #     bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, labels=labels
    # )
    save_name = path + "/comparision_acc_mac_{}".format(device)
    plt.savefig(save_name + args.save_ext + ".pdf", ext="pdf", bbox_inches="tight")
    plt.savefig(save_name + args.save_ext + ".png", ext="png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        nargs="+",
        type=str,
        default="mcts_generated/arch_profile_mcts_v7_t8_generated_cifar_macro_mcts_v7_sim_100_mcts_architecture_cpu_layers_imagenet_mcts_with_acc.csv",
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
    parser.add_argument(
        "-s",
        "--save_ext",
        type=str,
        default="mnist2_op_gap",
        help="additional name to pdf image results",
    )
    parser.add_argument("--line", action="store_true", default=False)
    args = parser.parse_args()
    list_files = args.filename
    if type(list_files) is list:
        for csv_file in list_files:
            draw_errorbar_graph(csv_file, args.device, args.path)
    else:
        draw_errorbar_graph(list_files, args.device, args.path)
