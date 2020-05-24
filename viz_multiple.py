import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np

from matplotlib.patches import Patch


def device_color_code(index):
    arr_color = ["green", "red", "black", "blue", "brown"]
    return arr_color[index]


def draw_errorbar_graph(index, file_name, path, subplot):
    df = pd.read_csv(file_name)
    print(df)
    print("result image can be seen in path ./{}".format(args.path))

    color = device_color_code(index)
    y = df["acc"]
    x = df["mean_lat"]
    e = df["std_dev_lat"]
    print(y)
    print(x)
    print(e)

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
        label=index,
        color=color,
    )


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
        "-p", "--path", type=str, default="img", help="path to pdf image results"
    )
    parser.add_argument(
        "-s",
        "--save_ext",
        type=str,
        default="multiple_mnist2_op_gap",
        help="additional name to pdf image results",
    )
    parser.add_argument("--line", action="store_true", default=False)
    args = parser.parse_args()
    list_files = args.filename
    subplot = plt.figure(figsize=(8, 5))
    figure, subplot = plt.subplots(nrows=1, ncols=1)

    subplot.set_title("MNIST Acc vs Latency Behaviour")
    subplot.set_ylabel("acc(%)", fontsize=13)
    subplot.set_xlabel("latency(ms)", fontsize=13)
    # labels = []
    if type(list_files) is list:
        for i, csv_file in enumerate(list_files):
            draw_errorbar_graph(i, csv_file, args.path, subplot)

    # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    save_name = args.path + "/acc_lat_"
    plt.savefig(save_name + args.save_ext + ".pdf", ext="pdf", bbox_inches="tight")
    plt.savefig(save_name + args.save_ext + ".png", ext="png", bbox_inches="tight")
    plt.show()
