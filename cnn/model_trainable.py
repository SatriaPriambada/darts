# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import train, test, get_data_loaders, ConvNet

from model import HeterogenousNetworkImageNet
from model import HeterogenousNetworkCIFAR
from model import HeterogenousNetworkMNIST
import logging

logger = logging.getLogger(__name__)

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class TrainMNIST(tune.Trainable):
    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
        )

    def _train(self):
        train(self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def convert_str_to_CIFAR_Network(
    name,
    selected_layers,
    cell_layers,
    none_layers,
    skip_conn,
    init_channels,
    layers,
    auxiliary,
    n_classes,
):

    return {
        "name": name,
        "cell_layers": cell_layers,
        "none_layers": none_layers,
        "skip_conn": skip_conn,
        "model": HeterogenousNetworkCIFAR(
            init_channels, n_classes, layers, auxiliary, selected_layers
        ),
    }


class TrainHeteroNetCIFAR(tune.Trainable):
    def _setup(self, config):
        print("[Tio]Start setup MODEL", flush=True)
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        print("[Tio]Start setup MODEL {}".format(config))
        model_conf = config["model_conf"]
        print("[Tio]New MODEL {}".format(model_conf))
        self.model = model_conf["model"]
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
        )

    def _train(self):
        train(self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def convert_str_to_ImageNet_Network(
    name,
    selected_layers,
    cell_layers,
    none_layers,
    skip_conn,
    init_channels,
    layers,
    auxiliary,
    n_classes,
):

    return {
        "name": name,
        "cell_layers": cell_layers,
        "none_layers": none_layers,
        "skip_conn": skip_conn,
        "model": HeterogenousNetworkImageNet(
            init_channels, n_classes, layers, auxiliary, selected_layers
        ),
    }
