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
OPORTUNITY_GAP_ARCHITECTURE = "mcts_generated/t8_generated_cifar_macro_mcts_v7_sim_100_mcts_architecture_cpu_layers.csv"
# Change these values if you want the training to run quicker or slower.
SKIN_CANCER_CLASSES = 7
INPUT_SIZE = 224
BATCH_SIZE = 128
ngpus_per_node = 1
best_acc1 = 0


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


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
        train_mnist_skin_cancer,
        scheduler=sched,
        config=hyperparameter_space,
        resources_per_trial={"gpu": ngpus_per_node},
        verbose=1,
        name="train_mnist_skin_cancer",  # This is used to specify the logging directory.
    )

    print("Finishing latency strata: {}".format(device_id))

    dfs = analysis.fetch_trial_dataframes()
    [d.mean_accuracy.plot() for d in dfs.values()]
    plt.xlabel("epoch")
    plt.ylabel("Test Accuracy")
    plt.savefig(
        path + "/train_mnist_skin_cancer_{}_{}.pdf".format(device, device_id),
        ext="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        path + "/train_mnist_skin_cancer_{}_{}.png".format(device, device_id),
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

    # to make the results are reproducible
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    data_dir = "~/Downloads/skin-cancer-mnist-ham10000/"
    all_image_path = glob(os.path.join(data_dir, "*", "*.jpg"))
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
    }
    lesion_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "dermatofibroma",
        "bkl": "Benign keratosis-like lesions ",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    norm_mean = [0.763038, 0.54564667, 0.57004464]
    norm_std = [0.14092727, 0.15261286, 0.1699712]

    df_original = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
    df_original["path"] = df_original["image_id"].map(imageid_path_dict.get)
    df_original["cell_type"] = df_original["dx"].map(lesion_type_dict.get)
    df_original["cell_type_idx"] = pd.Categorical(df_original["cell_type"]).codes
    print(df_original.head())

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby("lesion_id").count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup["image_id"] == 1]
    df_undup.reset_index(inplace=True)
    df_undup.head()

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup["lesion_id"])
        if x in unique_list:
            return "unduplicated"
        else:
            return "duplicated"

    # create a new colum that is a copy of the lesion_id column
    df_original["duplicates"] = df_original["lesion_id"]
    # apply the function to this new column
    df_original["duplicates"] = df_original["duplicates"].apply(get_duplicates)
    df_original.head()

    print(df_original["duplicates"].value_counts())

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original["duplicates"] == "unduplicated"]
    df_undup.shape

    # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup["cell_type_idx"]
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    df_val.shape

    df_val["cell_type_idx"].value_counts()

    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val["image_id"])
        if str(x) in val_list:
            return "val"
        else:
            return "train"

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original["train_or_val"] = df_original["image_id"]
    # apply the function to this new column
    df_original["train_or_val"] = df_original["train_or_val"].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original["train_or_val"] == "train"]
    print(len(df_train))
    print(len(df_val))

    print(df_train["cell_type_idx"].value_counts())
    print(df_val["cell_type"].value_counts())

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = df_train.append(
                [df_train.loc[df_train["cell_type_idx"] == i, :]]
                * (data_aug_rate[i] - 1),
                ignore_index=True,
            )
    df_train["cell_type"].value_counts()

    # # We can split the test set again in a validation set and a true test set:
    # df_val, df_test = train_test_split(df_val, test_size=0.5)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    # Data loading code
    normalize = transforms.Normalize(mean=norm_mean, std=norm_stds)
    train_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    # define the transformation of the val images.
    val_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = HAM10000(df_train, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Same for the validation set:
    validation_set = HAM10000(df_val, transform=train_transform)
    test_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_mnist_skin_cancer(config):
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
    criterion_smooth = CrossEntropyLabelSmooth(SKIN_CANCER_CLASSES, label_smooth)
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

    logfile.write("[Tio] acc {}".format(best_acc))
    logfile.flush()
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
    parser.add_argument(
        "--learning_rate", type=float, default=0.025, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")

    parser.add_argument(
        "--final_epochs", default=120, type=int, help="number of total epochs to run"
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

    num_classes = SKIN_CANCER_CLASSES
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
