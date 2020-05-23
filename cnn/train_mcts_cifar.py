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
from model import HeterogenousNetworkCIFAR

CIFAR_CLASSES = 10
best_acc = 0


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model_name = "Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.0', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 4), ('mobilenetV2_1.0', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('max_pool_3x3', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('mobilenetV3_1.0_3x3_Hswish', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_3x3', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('dil_conv_5x5', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('avg_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hswish', 0), ('max_pool_3x3', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('mobilenetV2_1.0', 3), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('skip_connect', 2)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.0', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 4), ('mobilenetV2_1.0', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('max_pool_3x3', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('mobilenetV3_1.0_3x3_Hswish', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_3x3', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('dil_conv_5x5', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('avg_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hswish', 0), ('max_pool_3x3', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('mobilenetV2_1.0', 3), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('skip_connect', 2)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish_SE', 0), ('mobilenetV2_1.4', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 3), ('max_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_7x7', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('max_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 2), ('mobilenetV3_1.0_3x3_Hsigmoid', 2), ('mobilenetV2_1.4', 4), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('sep_conv_5x5', 1), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('sep_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_5x5', 3), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('mobilenetV3_1.0_3x3_Hsigmoid', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('sep_conv_7x7', 1)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_7x7', 0), ('skip_connect', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_7x7', 2), ('mobilenetV3_1.0_5x5_Hswish', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 1), ('sep_conv_7x7', 3), ('sep_conv_3x3', 3), ('sep_conv_7x7', 3), ('mobilenetV3_1.0_5x5_Hsigmoid', 1)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.0', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 4), ('mobilenetV2_1.0', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('max_pool_3x3', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('mobilenetV3_1.0_3x3_Hswish', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_3x3', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('dil_conv_5x5', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('avg_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hswish', 0), ('max_pool_3x3', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('mobilenetV2_1.0', 3), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('skip_connect', 2)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish_SE', 0), ('mobilenetV2_1.4', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 3), ('max_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_7x7', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('max_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 2), ('mobilenetV3_1.0_3x3_Hsigmoid', 2), ('mobilenetV2_1.4', 4), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0)], reduce_concat=range(2, 6));Genotype(normal=[('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.4', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 0), ('mobilenetV2_1.0', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('sep_conv_7x7', 1)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hsigmoid', 0), ('mobilenetV3_1.0_3x3_Hswish', 1), ('mobilenetV2_1.4', 0), ('sep_conv_7x7', 0), ('mobilenetV3_1.0_5x5_Hswish_SE', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_3x3', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('dil_conv_5x5', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('avg_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hswish', 0), ('max_pool_3x3', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('mobilenetV2_1.0', 3), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('skip_connect', 2)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_3x3', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('dil_conv_5x5', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('avg_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hswish', 0), ('max_pool_3x3', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('mobilenetV2_1.0', 3), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 2), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('skip_connect', 2)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('mobilenetV3_1.0_3x3_Hswish', 0), ('sep_conv_3x3', 2), ('sep_conv_7x7', 1), ('dil_conv_5x5', 1), ('mobilenetV3_1.0_3x3_Hswish', 3), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.0', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 4), ('mobilenetV2_1.0', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('max_pool_3x3', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('mobilenetV3_1.0_3x3_Hswish', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hswish', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('mobilenetV2_1.0', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('sep_conv_5x5', 1), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('sep_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_5x5', 3), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 2), ('mobilenetV3_1.0_3x3_Hsigmoid', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 2), ('mobilenetV3_1.0_5x5_Hswish', 0), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('sep_conv_7x7', 1)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV2_1.0', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 4), ('mobilenetV2_1.0', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('mobilenetV3_1.0_5x5_Hsigmoid_SE', 1), ('max_pool_3x3', 0), ('mobilenetV3_1.0_3x3_Hsigmoid', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 3), ('mobilenetV3_1.0_3x3_Hswish', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6));Genotype(normal=[('mobilenetV3_1.0_5x5_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hsigmoid', 0), ('mobilenetV3_1.0_3x3_Hswish', 0), ('sep_conv_3x3', 2), ('sep_conv_7x7', 1), ('dil_conv_5x5', 1), ('mobilenetV3_1.0_3x3_Hswish', 3), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('mobilenetV3_1.0_5x5_Hswish_SE', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('mobilenetV3_1.0_5x5_Hsigmoid', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_7x7', 0), ('skip_connect', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_7x7', 2), ('mobilenetV3_1.0_5x5_Hswish', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 1), ('sep_conv_7x7', 3), ('sep_conv_3x3', 3), ('sep_conv_7x7', 3), ('mobilenetV3_1.0_5x5_Hsigmoid', 1)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_7x7', 0), ('skip_connect', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_7x7', 2), ('mobilenetV3_1.0_5x5_Hswish', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 1), ('sep_conv_7x7', 3), ('sep_conv_3x3', 3), ('sep_conv_7x7', 3), ('mobilenetV3_1.0_5x5_Hsigmoid', 1)], reduce_concat=range(2, 6));Genotype(normal=[('sep_conv_7x7', 0), ('skip_connect', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 0), ('mobilenetV3_1.0_5x5_Hswish', 2), ('sep_conv_7x7', 2), ('mobilenetV3_1.0_5x5_Hswish', 2), ('avg_pool_3x3', 3), ('mobilenetV3_1.0_5x5_Hswish', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('mobilenetV3_1.0_3x3_Hsigmoid_SE', 1), ('mobilenetV3_1.0_5x5_Hswish', 1), ('mobilenetV3_1.0_3x3_Hswish_SE', 1), ('sep_conv_7x7', 3), ('sep_conv_3x3', 3), ('sep_conv_7x7', 3), ('mobilenetV3_1.0_5x5_Hsigmoid', 1)], reduce_concat=range(2, 6))"

    selected_layers = model_name.split(";")
    model = HeterogenousNetworkCIFAR(
        args.init_channels, CIFAR_CLASSES, 25, args.auxiliary, selected_layers,
    )
    model.drop_path_prob = args.drop_path_prob
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.data, train=False, download=True, transform=valid_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)
    )
    logfile = open("log_mcts_cifar.txt", "w")
    logfile.write("[Tio] log for architecture id {}".format(model_name))
    device = torch.device("cuda")
    for epoch in range(args.epochs):
        scheduler.step()
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # Train model to get accuracy.
        logfile.write("start training epoch {} \n".format(epoch))
        logfile.flush()
        train_acc, loss = torch_1_v_4_train(
            epoch,
            model,
            optimizer,
            criterion,
            train_loader,
            logfile,
            device,
            args.auxiliary,
        )
        # Obtain validation accuracy.
        logfile.write("start test epoch {} \n".format(epoch))
        logfile.flush()
        acc, _ = torch_1_v_4_test(epoch, model, criterion, test_loader, logfile, device)
        logfile.write("[Tio] epoch {} acc {} loss {}".format(epoch, acc, loss))
        logfile.flush()
        tune.track.log(mean_accuracy=acc)
        torch.save(model, "./checkpoint_mcts_cifar.pth")
        if acc > best_acc:
            logfile.write(
                "[Tio] find new best model epoch {} acc {} loss {}".format(
                    epoch, acc, loss
                )
            )
            best_acc = acc
            torch.save(model, "./best_mcts_cifar.pth")
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
        logfile.write("epoch: {} batchid: {}\n".format(epoch, batch_idx))
        logfile.flush()

        data = Variable(data).to(device)
        target = Variable(target).to(device)
        logfile.write("data: {} target: {}\n".format(data, target))
        logfile.flush()
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
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument(
        "--data", type=str, default="~/data", help="location of the data corpus"
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
    parser.add_argument("--layers", type=int, default=20, help="total number of layers")
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
        "--arch", type=str, default="DARTS", help="which architecture to use"
    )
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
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
    fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    main()
