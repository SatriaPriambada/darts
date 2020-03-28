import tensorflow as tf
try:
    tf.get_logger().setLevel('INFO')
except Exception as exc:
    print(exc)
import warnings
warnings.simplefilter("ignore")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
from ray.tune.examples.mnist_pytorch import train, test, ConvNet, get_data_loaders
from model_trainable import TrainHeteroNetCIFAR
import ray
from ray import tune
from ray.tune import track

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("ggplot")
from na_scheduler import NAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler

from model import HeterogenousNetworkImageNet
from model import HeterogenousNetworkCIFAR
from model import HeterogenousNetworkMNIST

dset.MNIST("~/data", train=True, download=True)
dset.CIFAR10("~/data", train=True, download=True)
import asyncio
import sys
import time
import pandas as pd
from profile_macro_nn import connvert_df_to_list_arch
import async_timeout

OPORTUNITY_GAP_ARCHITECTURE = "test_arch.csv"
# OPORTUNITY_GAP_ARCHITECTURE = "generated_macro_architecture_cpu_layers.csv"
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

async def per_res_train(device, 
        device_id, 
        hyperparameter_space, 
        sched, 
        path):
    print('Running for GPU: {}'.format(device_id))
        
    analysis = tune.run(
        train_heterogenous_network_mnist, 
        scheduler=sched, 
        config=hyperparameter_space, 
        resources_per_trial={
            "gpu": 1
        },
        verbose=1,
        name="na_train_hetero_mnist"  # This is used to specify the logging directory.
    )

    print('Finishing GPU: {}'.format(device_id))

    dfs = analysis.fetch_trial_dataframes()
    [d.mean_accuracy.plot() for d in dfs.values()]
    plt.xlabel("epoch"); plt.ylabel("Test Accuracy"); 
    plt.savefig(path + '/train_mnist_{}_{}.pdf'.format(device, device_id), ext='pdf', bbox_inches='tight')
    plt.savefig(path + '/train_mnist_{}_{}.png'.format(device, device_id), ext='png', bbox_inches='tight')


async def async_train(device,
        hyperparameter_space,
        sched, 
        path):
    devices = [5]
    tasks = []

    for device_id in devices:
        task = asyncio.ensure_future(per_res_train(device, 
            device_id, 
            hyperparameter_space, 
            sched, 
            path)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

def train_mnist(config):
    model = ConvNet()
    logfile = open("log.txt","w")
    train_loader, test_loader = get_data_loaders()

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    for i in range(20):
        # Train for 1 epoch
        train(model, optimizer, train_loader)  
        acc = test(model, test_loader)  # Obtain validation accuracy.
        tune.track.log(mean_accuracy=acc)
        if i % 5 == 0:
            torch.save(model, "./{}.pth".format(config["model_name"])) # This saves the model to the trial directory
    logfile.close()

def train_heterogenous_network_mnist(config):
    model_name = config["architecture"]["name"]

    logfile = open("log.txt","w")
    logfile.write("[Tio] inside here Test Logging Info")
    #logfile.write("[Tio] training model {}".format(model_name))
    selected_layers = model_name.split(";")
    model = HeterogenousNetworkMNIST(
        config["architecture"]["init_channels"], 
        config["architecture"]["num_classes"],
        config["architecture"]["layers"], 
        config["architecture"]["auxiliary"],
        selected_layers
    )
    model.drop_path_prob = config["architecture"]["drop_path_prob"]
    
    train_loader, test_loader = get_data_loaders()

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    best_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(100):
        logfile.write("[Tio] epoch {}\n".format(epoch))
        logfile.flush()
        # Train model to get accuracy.
        logfile.write("start training epoch {} \n".format(epoch))
        logfile.flush()
        torch_1_v_4_train(epoch, model, optimizer, train_loader, logfile, device)
        # Obtain validation accuracy.
        logfile.write("start test epoch {} \n".format(epoch))
        logfile.flush()
        acc = torch_1_v_4_test(epoch, model, test_loader, logfile, device)  
        logfile.write("[Tio] acc {}".format(acc))
        logfile.flush()
        tune.track.log(mean_accuracy=acc)
        torch.save(model, "./checkpoint_{}.pth".format(config["architecture"]["id"]))
        if acc > best_acc:
            best_acc = acc
            torch.save(model, "./best_{}.pth".format(config["architecture"]["id"]))

    logfile.close()

def torch_1_v_4_train(epoch, model, optimizer, train_loader, logfile, device=torch.device("cpu")):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        logfile.flush()
        if batch_idx * len(data) > EPOCH_SIZE:
            break
        logfile.write("valid batch\n")
        logfile.flush()
        data, target = data.to(device), target.to(device)
        logfile.write("data {}, target {} \n".format(data, target))
        logfile.flush()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        logfile.write("loss {} \n".format(loss))
        logfile.flush()
        loss.backward()
        optimizer.step()
        logfile.write("optimizer step \n")
        logfile.flush()

def torch_1_v_4_test(epoch, model, test_loader, logfile, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    logfile.write("start test\n")
    logfile.flush()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    logfile.write("finish test\n")
    logfile.flush()
    return correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu-i7-4578U', help='device used for profile')
    parser.add_argument('-p', '--path', type=str, default='img', help='path to pdf image results')
    parser.add_argument('-l', '--layers', type=int, default=25, help='number of layers')
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed) 
    init_channels = 36
    layers = 25
    auxiliary = True
    drop_path_prob = 0.2
    num_classes = 10

    ray.shutdown()  
    ray.init(log_to_driver=False)
    df_op_gap = pd.read_csv(OPORTUNITY_GAP_ARCHITECTURE)
    sampled_architecture = connvert_df_to_list_arch(df_op_gap, 
        init_channels, layers, auxiliary, num_classes)

    model_architectures = [{
            "id": i,
            "name": arch['name'],
            "cell_layers": arch["cell_layers"],
            "none_layers": arch["none_layers"],
            "skip_conn": arch["skip_conn"],
            "init_channels": init_channels, 
            "layers": layers, 
            "auxiliary": auxiliary,
            "drop_path_prob": drop_path_prob,
            "num_classes": num_classes
        } 
        for i, arch in enumerate(sampled_architecture)
    ]

    hyperparameter_space = {
        "model_name": "train_mnist",
        "architecture": tune.grid_search(model_architectures),
        "lr": tune.grid_search([0.001]),
        "momentum":  tune.grid_search([0.9])
    }

    sched = NAScheduler(
        metric='mean_accuracy',
        mode="max",
        grace_period=1,
    )

    #start parallel async execution for each latency strata
    _start = time.time()
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(async_train(
        args.device,
        hyperparameter_space,
        sched, 
        args.path)
    )

    loop.run_until_complete(future)
    print(f"Execution time: { time.time() - _start }")
    loop.close()
