import torch
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from pathlib import Path

import time
import utils
from thop import profile
from statistics import stdev

from model import HeterogenousNetworkImageNet
from model_trainable import convert_str_to_ImageNet_Network
import acc_profiler
import latency_profiler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse

filename = "arch_profile_imagenet_v1"
OPORTUNITY_GAP_ARCHITECTURE = (
    "t1_generated_imagenet_macro_mcts_sim_100_mcts_architecture_layers.csv"
)
INPUT_BATCH = 1
INPUT_CHANNEL = 3
INPUT_SIZE = 224
IMAGENET_CLASSES = 1000


def connvert_df_to_list_arch(df, init_channels, layers, auxiliary, num_classes):
    architectures = []
    print(df)
    list_arch_name = df["name"].tolist()
    for i, name in enumerate(list_arch_name):
        selected_layers = name.split(";")
        cell_layers = df.iloc[i]["cell_layers"]
        none_layers = df.iloc[i]["none_layers"]
        skip_conn = df.iloc[i]["skip_conn"]
        architectures.append(
            convert_str_to_ImageNet_Network(
                name,
                selected_layers,
                cell_layers,
                none_layers,
                skip_conn,
                init_channels,
                layers,
                auxiliary,
                num_classes,
            )
        )
    return architectures


def profile_arch_lat_and_acc(
    dataset_name, test_loader, sampled_architectures, criterion, device, drop_path_prob
):
    dict_list = []
    if dataset_name == "imagenet":
        input = torch.zeros(INPUT_BATCH, INPUT_CHANNEL, INPUT_SIZE, INPUT_SIZE).to(
            device
        )
    else:
        sys.exit("Error!, dataset name not defined")
    print("start profiling")
    for architecture in sampled_architectures:
        model = architecture["model"].to(device)
        model.drop_path_prob = drop_path_prob
        # profile parameters
        # profile latencies
        try:
            macs, params = profile(model, inputs=(input,))
            mean_lat, latencies = latency_profiler.test_latency(model, input, device)
            dict_list.append(
                {
                    "name": architecture["name"],
                    "acc": 0,
                    "mean_lat": mean_lat,
                    "lat95": latencies[94],
                    "lat99": latencies[98],
                    "std_dev_lat": stdev(latencies),
                    "macs": macs,
                    "params": params,
                    "cell_layers": architecture["cell_layers"],
                    "none_layers": architecture["none_layers"],
                    "skip_conn": architecture["skip_conn"],
                }
            )
   
            # print(architecture)
            print("============================")        
        except RuntimeError as e:
            print("error forward pass:", e)        
# profile accuracy
        # valid_acc, valid_obj = acc_profiler.infer(test_loader, model, criterion, device)
        print("============================AAAA")

    model_df_with_acc_and_lat = pd.DataFrame.from_dict(dict_list)

    return model_df_with_acc_and_lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-l", "--layers", type=int, default=25, help="number of layers")
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)
    init_channels = 48
    layers = args.layers
    auxiliary = True
    drop_path_prob = 0
    dataset_name = "imagenet"
    num_classes = IMAGENET_CLASSES
    filepath = "/srv/data/datasets/ImageNet"
    if "gpu" in args.device:
        print("profile on gpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("profile on cpu")
        device = torch.device("cpu")

    print("Load Data from: {}".format(filepath))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    validdir = filepath + "/val"
    test_data = dset.ImageFolder(
        validdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=True, num_workers=4
    )

    df_op_gap = pd.read_csv("mcts_generated/" + OPORTUNITY_GAP_ARCHITECTURE)
    sampled_architecture = connvert_df_to_list_arch(
        df_op_gap, init_channels, layers, auxiliary, IMAGENET_CLASSES
    )

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = False
        torch.cuda.manual_seed(seed)
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        print("no gpu device available, use CPU")

    model_df_with_acc_and_lat = profile_arch_lat_and_acc(
        dataset_name,
        test_loader,
        sampled_architecture,
        criterion,
        device,
        drop_path_prob,
    )
    model_df_with_acc_and_lat.to_csv(
        Path("mcts_generated/" + filename + "_" + OPORTUNITY_GAP_ARCHITECTURE),
        index=None,
    )
