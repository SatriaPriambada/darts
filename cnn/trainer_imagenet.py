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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim

import ray
from ray import tune
from ray.tune import track

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("ggplot")
from na_scheduler import NAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler

from model import HeterogenousNetworkImageNet

import asyncio
import sys
import time
import pandas as pd
from profile_macro_nn import connvert_df_to_list_arch
import async_timeout

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
OPORTUNITY_GAP_ARCHITECTURE = (
    "generated_cifar_macro_mcts_mcts_architecture_cpu_layers.csv"
)
# Change these values if you want the training to run quicker or slower.
IMAGENET_CLASSES = 1000
ngpus_per_node = 1
best_acc1 = 0


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


async def per_res_train(device, device_id, hyperparameter_space, sched, path):
    print("Running for latency strata: {}".format(device_id))

    analysis = tune.run(
        train_heterogenous_network_imagenet,
        scheduler=sched,
        config=hyperparameter_space,
        resources_per_trial={"gpu": ngpus_per_node},
        verbose=1,
        name="train_heterogenous_network_imagenet",  # This is used to specify the logging directory.
    )

    print("Finishing latency strata: {}".format(device_id))

    dfs = analysis.fetch_trial_dataframes()
    [d.mean_accuracy.plot() for d in dfs.values()]
    plt.xlabel("epoch")
    plt.ylabel("Test Accuracy")
    plt.savefig(
        path + "/train_imagenet_{}_{}.pdf".format(device, device_id),
        ext="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        path + "/train_imagenet_{}_{}.png".format(device, device_id),
        ext="png",
        bbox_inches="tight",
    )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def async_train(device, model_architectures, n_family, sched, path):
    fam_member = int(len(model_architectures) / n_family)
    if fam_member == 0:
        fam_member = 1
    # print("possible family candidates per strata", fam_member)
    arr_of_models = list(chunks(model_architectures, fam_member))
    # print(arr_of_models)
    # print("len arr ", len(arr_of_models))

    tasks = []

    for device_id, arr_model in enumerate(arr_of_models):
        hyperparameter_space = {
            "model_name": "train_mnist",
            "architecture": tune.grid_search(arr_model),
            "lr": tune.grid_search([args.learning_rate]),
            "momentum": tune.grid_search([args.momentum]),
        }
        task = asyncio.ensure_future(
            per_res_train(device, device_id, hyperparameter_space, sched, path)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


def get_dist_data_loaders(batch_size, workers):
    print("==> Preparing data..")
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_heterogenous_network_imagenet(config):
    #    ngpus_per_node = torch.cuda.device_count()

    world_size = ngpus_per_node * args.world_size

    model_name = config["architecture"]["name"]

    selected_layers = model_name.split(";")
    model = HeterogenousNetworkImageNet(
        config["architecture"]["init_channels"],
        config["architecture"]["num_classes"],
        config["architecture"]["layers"],
        config["architecture"]["auxiliary"],
        selected_layers,
    )
    model.drop_path_prob = config["architecture"]["drop_path_prob"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    gamma = 0.97
    decay_period = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_period, gamma=gamma)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    label_smooth = 0.1
    criterion_smooth = CrossEntropyLabelSmooth(IMAGENET_CLASSES, label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    valid_port = args.dist_port + int(config["architecture"]["id"])
    full_url = args.dist_url + ":" + str(valid_port)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=full_url,
        rank=args.rank,
        world_size=world_size,
    )
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True
    )
    # Train model to get accuracy.

    logfile = open("log.txt", "w")
    logfile.write(
        "[Tio] log for architecture id {}".format(config["architecture"]["id"])
    )
    # logfile.write("[Tio] training model {}".format(model_name))
    train_loader, test_loader = get_dist_data_loaders(args.batch_size, args.workers)

    best_acc = 0
    for epoch in range(args.final_epochs):
        scheduler.step()
        model.drop_path_prob = args.drop_path_prob * epoch / args.final_epochs
        # Train model to get accuracy.
        logfile.write("start training epoch {} \n".format(epoch))
        logfile.flush()
        train_acc, _ = torch_1_v_4_train(
            epoch,
            model,
            optimizer,
            criterion_smooth,
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

        data = Variable(data).to(device)
        target = Variable(target).to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(data)
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += auxiliary_weight * loss_aux
        loss.backward()
        logfile.write("loss {} \n".format(loss))
        logfile.flush()
        grad_clip = 5
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = data.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        logfile.write(
            "train {} {} {} {}".format(batch_idx, objs.avg, top1.avg, top5.avg)
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
                "valid {} {} {} {}".format(batch_idx, objs.avg, top1.avg, top5.avg)
            )
            logfile.flush()
    return top1.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imagenet distributed training models")
    #    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                        choices=model_names,
    #                        help='model architecture: ' +
    #                            ' | '.join(model_names) +
    #                            ' (default: resnet18)')

    parser.add_argument(
        "--learning_rate", type=float, default=0.025, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")

    parser.add_argument(
        "--final_epochs", default=90, type=int, help="number of total epochs to run"
    )

    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--workers", type=int, default=4, help="")
    parser.add_argument(
        "--gpu_devices", type=int, nargs="+", default=[5, 6, 7], help=""
    )

    parser.add_argument(
        "--data",
        default="/srv/data/datasets/ImageNet",
        type=str,
        help="path to ImageNet data",
    )
    parser.add_argument(
        "-print",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--dist-url", default="tcp://127.0.0.1", type=str, help="")
    parser.add_argument("--dist-port", default=3456, type=int, help="")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="")
    parser.add_argument("--rank", default=0, type=int, help="")
    parser.add_argument("--world_size", default=1, type=int, help="")
    parser.add_argument("--distributed", action="store_true", help="")
    args = parser.parse_args()

    gpu_devices = ",".join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    parser.add_argument(
        "--report_freq", type=float, default=50, help="report frequency"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
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
        default="cpu-i7-4578U",
        help="device used for profile",
    )
    parser.add_argument(
        "-p", "--path", type=str, default="img", help="path to pdf image results"
    )

    parser.add_argument(
        "--parallel", action="store_true", default=False, help="data parallelism"
    )

    args = parser.parse_args()

    num_classes = IMAGENET_CLASSES
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
