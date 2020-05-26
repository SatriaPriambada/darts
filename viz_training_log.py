import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib.patches import Patch
from pathlib import Path

import os


def update_df_acc(file_name, path):
    df = pd.DataFrame()
    with open(file_name, "r") as f:
        lines = f.readlines()
        # print(text)
        for line in lines:
            if "[Tio] epoch" in line:
                # print(line)
                epoch = int(line.split()[2])
                acc = float(line.split()[4])
                loss = float(line.split()[6])
                # print({"epoch": epoch, "acc": acc, "loss": loss})
                df = df.append(
                    {"epoch": epoch, "acc": acc, "loss": loss}, ignore_index=True
                )
    print(df)
    df.to_csv(Path("train_log.csv"), index=None)

    figure, axes = plt.subplots(nrows=2, ncols=1)
    y = df["acc"]
    x = df["epoch"]
    axes[0].plot(x, y)
    axes[0].set_title("Training Log Acc")
    axes[0].set_ylabel("acc(%)", fontsize=13)
    axes[0].set_xlabel("epoch", fontsize=13)

    locs = axes[0].get_xticks()
    print(locs)
    x_ticklabels = [str(i) for i in locs]
    axes[0].set_xticklabels(x_ticklabels)

    y = df["loss"]
    x = df["epoch"]
    axes[1].plot(x, y, color="green")
    axes[1].set_title("Training Log Loss")
    axes[1].set_ylabel("Loss", fontsize=13)
    axes[1].set_xlabel("epoch", fontsize=13)

    locs = axes[1].get_xticks()
    print(locs)
    x_ticklabels = [str(i) for i in locs]
    axes[1].set_xticklabels(x_ticklabels)

    figure.tight_layout(pad=0.3)

    # lgd = plt.legend(
    #     bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, labels=labels
    # )
    save_name = path + "/training_log"
    plt.savefig(save_name + ".pdf", ext="pdf", bbox_inches="tight")
    plt.savefig(save_name + ".png", ext="png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="log_mcts_cifar.txt",
        help="log file name",
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
    print(list_files)

    update_df_acc(list_files, args.imgpath)
