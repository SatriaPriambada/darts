import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
import torch
import torch.nn.functional as F
import torch.optim as optim

import ray
from ray import tune
from ray.tune import track

import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("ggplot")
from na_scheduler import NAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler

from model import HeterogenousNetworkCIFAR

dset.CIFAR100("~/data", train=True, download=True)
import asyncio
import sys
import time
import pandas as pd
from profile_macro_nn import connvert_df_to_list_arch
import async_timeout

OPORTUNITY_GAP_ARCHITECTURE = "mcts_generated/arch_op_gap_cifar100.csv"
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 256


async def per_res_train(device, device_id, hyperparameter_space, sched, path):
    print("Running for latency strata: {}".format(device_id))

    analysis = tune.run(
        train_heterogenous_network_cifar,
        scheduler=sched,
        config=hyperparameter_space,
        resources_per_trial={"gpu": 1},
        verbose=1,
        name="train_mcts_cifar100",  # This is used to specify the logging directory.
    )

    print("Finishing latency strata: {}".format(device_id))

    dfs = analysis.fetch_trial_dataframes()
    [d.mean_accuracy.plot() for d in dfs.values()]
    plt.xlabel("epoch")
    plt.ylabel("Test Accuracy")
    plt.savefig(
        path + "/train_cifar100_{}_{}.pdf".format(device, device_id),
        ext="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        path + "/train_cifar100_{}_{}.png".format(device, device_id),
        ext="png",
        bbox_inches="tight",
    )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def async_train(device, model_architectures, n_family, sched, path):

    arr_of_models = list(chunks(model_architectures, len(model_architectures)))
    print(arr_of_models)
    devices = [i for i in range(n_family)]
    tasks = []

    for device_id in devices:
        hyperparameter_space = {
            "model_name": "train_mnist",
            "architecture": tune.grid_search(arr_of_models[device_id]),
            "lr": tune.grid_search([args.learning_rate]),
            "momentum": tune.grid_search([args.momentum]),
        }
        task = asyncio.ensure_future(
            per_res_train(device, device_id, hyperparameter_space, sched, path)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


CIFAR_CLASSES = 100


def get_data_loaders(batch_size, workers, args):
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(
        root="~/data", train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR100(
        root="~/data", train=False, download=True, transform=valid_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
    )

    test_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=workers,
    )
    return train_loader, test_loader


def train_heterogenous_network_cifar(config):
    model_name = config["architecture"]["name"]

    logfile = open("log.txt", "w")
    logfile.write(
        "[Tio] log for architecture id {}".format(config["architecture"]["id"])
    )
    # logfile.write("[Tio] training model {}".format(model_name))
    selected_layers = model_name.split(";")
    model = HeterogenousNetworkCIFAR(
        config["architecture"]["init_channels"],
        config["architecture"]["num_classes"],
        config["architecture"]["layers"],
        config["architecture"]["auxiliary"],
        selected_layers,
    )
    model.drop_path_prob = config["architecture"]["drop_path_prob"]
    workers = 4
    batch_size = EPOCH_SIZE
    if config["architecture"]["cell_layers"] > 18:
        batch_size = 64
    elif config["architecture"]["cell_layers"] > 12:
        batch_size = 128

    train_loader, test_loader = get_data_loaders(batch_size, workers, args)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)
    )
    best_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    for epoch in range(50):
        scheduler.step()
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # Train model to get accuracy.
        logfile.write("start training epoch {} \n".format(epoch))
        logfile.flush()
        train_acc, _ = torch_1_v_4_train(
            epoch,
            model,
            optimizer,
            criterion,
            train_loader,
            logfile,
            device,
            config["architecture"]["auxiliary"],
        )
        # Obtain validation accuracy.
        logfile.write("start test epoch {} \n".format(epoch))
        logfile.flush()
        acc, _ = torch_1_v_4_test(epoch, model, criterion, test_loader, logfile, device)
        logfile.write("[Tio] acc {}".format(acc))
        logfile.flush()
        tune.track.log(mean_accuracy=acc)
        torch.save(model, "./checkpoint_{}.pth".format(config["architecture"]["id"]))
        if acc > best_acc:
            best_acc = acc
            torch.save(model, "./best_{}.pth".format(config["architecture"]["id"]))

    logfile.close()


def torch_1_v_4_train(
    epoch,
    model,
    optimizer,
    criterion,
    train_loader,
    logfile,
    device=torch.device("cpu"),
    auxiliary=True,
):
    model.to(device)
    model.train()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        logfile.write("epoch: {} batchid: {}\n".format(epoch, batch_idx))
        logfile.flush()

        data = Variable(data).to(device)
        target = Variable(target).to(device)
        logfile.write("data: {} target: {}\n".format(data, target))
        logfile.flush()
        optimizer.zero_grad()
        logits, logits_aux = model(data)
        logfile.write("logits: {} logits_aux: {}\n".format(logits, logits_aux))
        logfile.flush()
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += auxiliary_weight * loss_aux
        loss.backward()
        logfile.write("loss {} \n".format(loss))
        logfile.flush()
        grad_clip = 5
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = data.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        logfile.write(
            "train %03d %e %f %f".format(batch_idx, objs.avg, top1.avg, top5.avg)
        )
        logfile.flush()

    return top1.avg, objs.avg


def torch_1_v_4_test(
    epoch, model, criterion, test_loader, logfile, device=torch.device("cpu")
):
    model.to(device)
    model.eval()
    logfile.write("start test\n")
    logfile.flush()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = Variable(data).to(device)
            target = Variable(target).to(device)

            logits, _ = model(data)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            logfile.write(
                "valid %03d %e %f %f".format(batch_idx, objs.avg, top1.avg, top5.avg)
            )
            logfile.flush()
    return top1.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("trainer_cifar.py")
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.025, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument(
        "--report_freq", type=float, default=50, help="report frequency"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--epochs", type=int, default=600, help="num of training epochs"
    )
    parser.add_argument(
        "--init_channels", type=int, default=36, help="num of init channels"
    )
    parser.add_argument("--layers", type=int, default=25, help="total number of layers")
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument(
        "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
    )
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss"
    )
    parser.add_argument(
        "--cutout", action="store_true", default=False, help="use cutout"
    )
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument(
        "--drop_path_prob", type=float, default=0.2, help="drop path probability"
    )
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
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
    args = parser.parse_args()

    args.save = "eval-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    num_classes = CIFAR_CLASSES
    ray.shutdown()
    ray.init(log_to_driver=False)
    df_op_gap = pd.read_csv(OPORTUNITY_GAP_ARCHITECTURE)
    sampled_architecture = connvert_df_to_list_arch(
        df_op_gap, args.init_channels, args.layers, args.auxiliary, num_classes
    )

    model_architectures = [
        {
            "id": i,
            "name": arch["name"],
            "cell_layers": arch["cell_layers"],
            "none_layers": arch["none_layers"],
            "skip_conn": arch["skip_conn"],
            "init_channels": args.init_channels,
            "layers": args.layers,
            "auxiliary": args.auxiliary,
            "drop_path_prob": args.drop_path_prob,
            "num_classes": num_classes,
        }
        for i, arch in enumerate(sampled_architecture)
    ]

    n_family = 1

    sched = NAScheduler(metric="mean_accuracy", mode="max", grace_period=1,)

    # start parallel async execution for each latency strata
    _start = time.time()
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(
        async_train(args.device, model_architectures, n_family, sched, args.path)
    )

    loop.run_until_complete(future)
    print(f"Execution time: { time.time() - _start }")
    loop.close()
