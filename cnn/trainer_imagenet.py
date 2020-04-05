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
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
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

OPORTUNITY_GAP_ARCHITECTURE = "test_arch.csv"
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 8
NUM_WORKERS = 4
IMAGENET_CLASSES = 1000
ngpus_per_node = 1


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


async def per_res_train(device, 
        device_id, 
        hyperparameter_space, 
        sched, 
        path):
    print('Running for latency strata: {}'.format(device_id))
    
    analysis = tune.run(
        train_heterogenous_network_imagenet, 
        scheduler=sched, 
        config=hyperparameter_space, 
        resources_per_trial={
            "gpu": ngpus_per_node
        },
        verbose=1,
        name="train_heterogenous_network_imagenet"  # This is used to specify the logging directory.
    )

    print('Finishing latency strata: {}'.format(device_id))

    dfs = analysis.fetch_trial_dataframes()
    [d.mean_accuracy.plot() for d in dfs.values()]
    plt.xlabel("epoch"); plt.ylabel("Test Accuracy"); 
    plt.savefig(path + '/train_imagenet_{}_{}.pdf'.format(device, device_id), ext='pdf', bbox_inches='tight')
    plt.savefig(path + '/train_imagenet_{}_{}.png'.format(device, device_id), ext='png', bbox_inches='tight')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def async_train(device,
        model_architectures,
        n_family,
        sched, 
        path):

      
    arr_of_models = list(chunks(model_architectures, len(model_architectures)))
    print(arr_of_models)
    devices = [i for i in range(n_family)]
    tasks = []

    for device_id in devices:
        hyperparameter_space = {
            "model_name": "train_mnist",
            "architecture": tune.grid_search(arr_of_models[device_id]),
            "lr": tune.grid_search([args.learning_rate]),
            "momentum":  tune.grid_search([args.momentum])
        }
        task = asyncio.ensure_future(per_res_train(device, 
            device_id, 
            hyperparameter_space, 
            sched, 
            path)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

def get_dist_data_loaders(batch_size, num_workers):
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    
    train_data = dset.CIFAR10(root="~/data", train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root="~/data", train=False, download=True, transform=valid_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    num_replicas=1)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        pin_memory=True, num_workers=num_workers, sampler=train_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,
                                                                    num_replicas=1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, 
        pin_memory=True, num_workers=num_workers, sampler=test_sampler)
    return train_loader, test_loader

def train_heterogenous_network_imagenet(config):
    model_name = config["architecture"]["name"]

    selected_layers = model_name.split(";")
    model = HeterogenousNetworkImageNet(
        config["architecture"]["init_channels"], 
        config["architecture"]["num_classes"],
        config["architecture"]["layers"], 
        config["architecture"]["auxiliary"],
        selected_layers
    )
    model.drop_path_prob = config["architecture"]["drop_path_prob"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    
    gamma = 0.97
    decay_period = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_period, gamma=gamma)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    label_smooth = 0.1
    criterion_smooth = CrossEntropyLabelSmooth(IMAGENET_CLASSES, label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    
    port = 3456 + int(config["architecture"]["id"])
    dist_url = 'tcp://127.0.0.1:' + str(port)
    dist_backend = 'nccl'
    rank = 0
    world_size= 1
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(model,))
        
def main_worker(model):
    model = torch.nn.parallel.DistributedDataParallel(model)
    # Train model to get accuracy.

    logfile = open("log.txt","w")
    logfile.write("[Tio] log for architecture id {}".format(config["architecture"]["id"]))
    #logfile.write("[Tio] training model {}".format(model_name))
    train_loader, test_loader = get_dist_data_loaders(BATCH_SIZE, NUM_WORKERS)

    best_acc = 0
    for epoch in range(args.epoch):
      scheduler.step()
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
      # Train model to get accuracy.
      logfile.write("start training epoch {} \n".format(epoch))
      logfile.flush()
      train_acc, _ = torch_1_v_4_train(epoch, model, optimizer, criterion, train_loader, logfile, device, config["architecture"]["auxiliary"])
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

def torch_1_v_4_train(epoch, model, optimizer, criterion, train_loader, logfile, device=torch.device("cpu"), auxiliary=True):
    model.to(device)
    model.train()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
      logfile.write("epoch: {} batchid: {}\n".format(epoch, batch_idx))
      logfile.flush()
      
      data = Variable(data).to(device)
      target = Variable(target).to(device)

      optimizer.zero_grad()
      logits, logits_aux = model(data)
      loss = criterion(logits, target)
      if auxiliary:
        loss_aux = criterion(logits_aux, target)
        loss += auxiliary_weight*loss_aux
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
      logfile.write("train %03d %e %f %f".format(batch_idx, objs.avg, top1.avg, top5.avg))
      logfile.flush()

    return top1.avg, objs.avg

def torch_1_v_4_test(epoch, model, criterion, test_loader, logfile, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    logfile.write("start test\n")
    logfile.flush()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
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

            logfile.write('valid %03d %e %f %f'.format(batch_idx, objs.avg, top1.avg, top5.avg))
            logfile.flush()
    return top1.avg, objs.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=25, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-d', '--device', type=str, default='cpu-i7-4578U', help='device used for profile')
    parser.add_argument('-p', '--path', type=str, default='img', help='path to pdf image results')

    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')

    args = parser.parse_args()

    num_classes = IMAGENET_CLASSES
    ray.shutdown()  
    ray.init(log_to_driver=False)
    df_op_gap = pd.read_csv(OPORTUNITY_GAP_ARCHITECTURE)
    sampled_architecture = connvert_df_to_list_arch(df_op_gap, 
        args.init_channels, args.layers, args.auxiliary, num_classes)

    model_architectures = [{
            "id": i,
            "name": arch['name'],
            "cell_layers": arch["cell_layers"],
            "none_layers": arch["none_layers"],
            "skip_conn": arch["skip_conn"],
            "init_channels": args.init_channels, 
            "layers": args.layers, 
            "auxiliary": args.auxiliary,
            "drop_path_prob": args.drop_path_prob,
            "num_classes": num_classes
        } 
        for i, arch in enumerate(sampled_architecture)
    ]

    n_family = 2

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
        model_architectures,
        n_family,
        sched, 
        args.path)
    )

    loop.run_until_complete(future)
    print(f"Execution time: { time.time() - _start }")
    loop.close()
