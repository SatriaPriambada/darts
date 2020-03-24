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

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

load_filename = "generated_micro"
filename = "generated_macro"
NFAMILY = 8
CIFAR_CLASSES = 10
NUM_SAMPLE = 125
CLUSTERS = 8

def generate_macro(dataset_name, 
                  test_loader,
                  seed, 
                  init_channels, 
                  layers,
                  n_family, 
                  auxiliary, 
                  drop_path_prob, 
                  device):

  np.random.seed(seed)
  
  criterion = nn.CrossEntropyLoss()
  if torch.cuda.is_available():
    logging.info('gpu device available')
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    criterion = nn.CrossEntropyLoss().cuda()
  else:
    logging.info('no gpu device available, use CPU')
    
  macro_network = MacroNetwork(init_channels, CIFAR_CLASSES, 
    layers, criterion, device=device)
  micro_genotypes = pd.read_csv(load_filename + '_{}_center.csv'.format(str(device)))
  print(micro_genotypes) 
  sampled_architecture = macro_network.sample_architecture(dataset_name, NUM_SAMPLE, 
    micro_genotypes, init_channels, layers, n_family, auxiliary)
  df_arch = pd.DataFrame(sampled_architecture)
  df_arch = df_arch.drop(columns=['model'])
  print(df_arch)
  df_arch.to_csv(Path(filename + '_architecture_{}_layers.csv'.format(str(device), str(args.layers))), 
   index = None)

  return df_arch

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--layers', type=int, default=25, help='number of layers')
  args = parser.parse_args()

  seed = 0
  init_channels = 36
  layers = args.layers
  auxiliary = True
  drop_path_prob = 0.2
  data_path = '~/data/'
  dataset_name = 'cifar10'
  num_classes = 10
  filepath = "~/data/" + dataset_name
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Load Data from: {}".format(filepath))
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  test_data = dset.CIFAR10(root='~/data/cifar10', download=True, transform=valid_transform)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=256,
                                            shuffle=True, num_workers=4)

  list_genotypes = generate_macro(
    dataset_name,
    test_loader,
    seed, 
    init_channels, 
    layers, 
    NFAMILY,
    auxiliary,    
    drop_path_prob, 
    device) 
