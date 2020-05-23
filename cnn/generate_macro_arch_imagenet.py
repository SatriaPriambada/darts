import os
import sys
import glob

import logging

import torch
import torch.nn as nn
import torch.utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import numpy as np
import pandas as pd

import time
import utils

import genotypes
from model import NetworkCIFAR
from macro_model_search import MacroNetwork
from architect import Architect
from profile_macro_nn import profile_arch_lat_and_acc
from pathlib import Path
import argparse

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

load_filename = "generated_micro_imagenet_center.csv"
filename = "t3_generated_imagenet_macro_mcts_sim_100"
NFAMILY = 8
CLASSES = 100
NUM_SAMPLE = 125
CLUSTER = 14

def generate_macro(
    dataset_name,
    test_loader,
    seed,
    init_channels,
    layers,
    n_family,
    auxiliary,
    drop_path_prob,
    device,
):
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        logging.info("gpu device available")
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        logging.info("no gpu device available, use CPU")

    macro_network = MacroNetwork(
        init_channels, CLASSES, layers, criterion, device=device
    )
    micro_genotypes = pd.read_csv(load_filename)
    # print(micro_genotypes)
    sampled_architecture = macro_network.sample_mcts_architecture(
        dataset_name,
        NUM_SAMPLE,
        micro_genotypes,
        init_channels,
        layers,
        n_family,
        CLUSTER,
        auxiliary,
        drop_path_prob,
    )
    df_arch = pd.DataFrame(sampled_architecture)
    df_arch = df_arch.drop(columns=["model"])
    print(df_arch)
    df_arch.to_csv(
        Path(
            filename
            + "_mcts_architecture_{}_layers.csv".format(str(device), str(args.layers))
        ),
        index=None,
    )

    return df_arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", type=int, default=25, help="number of layers")
    args = parser.parse_args()

    seed = 0
    init_channels = 36
    layers = args.layers
    auxiliary = False
    drop_path_prob = 0.2
    data_path = "~/data/"
    dataset_name = "imagenet"
    num_classes = 1000
    filepath = "/srv/data/datasets/ImageNet"
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    print("Load Data from: {}".format(filepath))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    valdir = os.path.join(filepath, "val")
    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    list_genotypes = generate_macro(
        dataset_name,
        test_loader,
        seed,
        init_channels,
        layers,
        NFAMILY,
        auxiliary,
        drop_path_prob,
        device,
    )
