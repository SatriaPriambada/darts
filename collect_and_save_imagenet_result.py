import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib.patches import Patch
from pathlib import Path

import os


def device_color_code(device):
    if device == "cpu-i7-4578U":
        return "red"
    elif device == "gpu-rtx2080":
        return "green"
    elif device == "gpu-v100":
        return "blue"
    elif device == "cpu-x6132":
        return "brown"


def draw_acc_subplot(df, subplot, color):
    y = df["acc"]
    x = df["mean_lat"]
    e = df["std_dev_lat"]
    subplot.errorbar(
        x,
        y,
        xerr=e,
        fmt=".",
        mfc=color,
        mec=color,
        linestyle="",
        ecolor=[color],
        label=args.device,
        color=color,
    )
    subplot.set_title("{} Params vs Latency Behaviour".format(args.device))
    subplot.set_ylabel("acc(%)", fontsize=13)
    subplot.set_xlabel("latency(ms)", fontsize=13)

    locs = subplot.get_xticks()
    print(locs)
    x_ticklabels = [str(i) for i in locs]
    subplot.set_xticklabels(x_ticklabels)


def update_df_acc(file_name, device, path):
    df = pd.read_csv(file_name)
    folder_path = "~/ray_results/train_mcts_imagenet"
    folder_path = os.path.expanduser(folder_path)
    print("folder_path: {}".format(folder_path))
    acc = 0
    row_id = 0
    for dirName, subdirList, fileList in os.walk(folder_path):
        for filename in fileList:
            if filename == "log.txt":
                print("find file {}".format(filename))
                with open(dirName + "/" + filename, "r") as f:
                    text = f.readlines()
                    last_line = text[-1]
                    split_acc = last_line.split(" ")
                    if len(split_acc) > 2:
                        top1acc = split_acc[-2]
                        print("FIND on log ", top1acc)
                        acc = float(top1acc)
            if "pth" in filename:
                # print('find file {}'.format(filename))
                m = re.match("(.*)\_(.*)\.(pth)", filename)
                row_id = int(m.group(2))
        # print('update row: {} acc: {}'.format(row_id, acc))
        if row_id < len(df.index):
            df.loc[row_id, ["acc"]] = acc
            # print(df.loc[df.index == row_id])
    nm = file_name.split(".")
    df.to_csv(Path(nm[0] + "_imagenet_mcts_with_acc.csv"), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        nargs="+",
        type=str,
        default="mcts_generated/arch_profile_mcts_v7_t8_generated_cifar_macro_mcts_v7_sim_100_mcts_architecture_cpu_layers.csv",
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
        "-p",
        "--imgpath",
        type=str,
        default="result_family_models",
        help="path to pdf image results",
    )
    args = parser.parse_args()
    list_files = args.filename

    if type(list_files) is list:
        for csv_file in list_files:
            update_df_acc(csv_file, args.device, args.imgpath)
    else:
        update_df_acc(list_files, args.device, args.imgpath)
