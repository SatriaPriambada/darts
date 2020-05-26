import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import HeterogenousNetworkImageNet
from pathlib import Path
import pandas as pd

# python cnn/test_imagenet_hetero.py --gpu 2 /srv/data/datasets/ImageNet
micro_medioid_filename = "generated_micro_imagenet_center.csv"
macro_generated_filename = "mcts_generated/t1_generated_imagenet_short_small.csv"

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=0, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=2, type=int, help="GPU id to use.")

# DARTS config
parser.add_argument(
    "--init_channels", type=int, default=48, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=14, help="total number of layers")
parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
parser.add_argument(
    "--model_path", type=str, default="EXP/model.pt", help="path of pretrained model"
)

best_acc1 = 0
CLASSES = 1000


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


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # Simply call main_worker function
    main_worker(args.gpu, 1, args)


def load_models(args):
    df_macro_arch = pd.read_csv(macro_generated_filename)
    df_micro_genotypes = pd.read_csv(micro_medioid_filename)

    selected_archs = []
    for selected_med in df_macro_arch["selected_medioid_idx"]:
        # print("selected_med", type(selected_med))
        selected_med = eval(selected_med)
        # print("selected_med after eval",type(selected_med))
        selected_genotype = [df_micro_genotypes.iloc[x, 0] for x in selected_med]
        model = HeterogenousNetworkImageNet(
            args.init_channels, 1000, 25, True, selected_genotype
        )
        model.drop_path_prob = 0

        torch.cuda.set_device(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
        model = model.cuda(args.gpu)
        model.load_state_dict(torch.load(args.model_path)["state_dict"])
        selected_archs.append(model)
    return selected_archs


def load_data(args):

    # Data loading code
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    val_loader = torch.utils.data.DataLoader(
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return val_loader


def note_acc_loss_for_drawing_graph(model_id, epoch, acc1, loss, df):
    df = df.append(
        {"model_id": model_id, "epoch": epoch, "acc1": acc1, "loss": loss},
        ignore_index=True,
    )
    df.to_csv(Path("acc_loss_log_model_{}.csv".format(model_id)), index=None)
    return df


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    # create model
    models = generate_models(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda(args.gpu)

    cudnn.benchmark = True
    val_loader, train_sampler = load_data(args)
    df = pd.DataFrame()
    for model_id, model in enumerate(models):
        # evaluate on validation set
        acc1, loss = validate(val_loader, model, criterion, args)
        df = note_acc_loss_for_drawing_graph(model_id, epoch, acc1.item(), loss, df)

    print("[Tio] finish training model-", model_id)
    print("[Tio] model-", model_id, " best_acc1:", best_acc1)
    best_acc1 = 0
    print("[Tio] reset best_acc1:", best_acc1)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, logits_aux = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
