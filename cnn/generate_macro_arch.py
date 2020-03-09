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

from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

import time
import utils
from thop import profile
from statistics import stdev


import genotypes
from model import NetworkCIFAR
from macro_model_search import MacroNetwork
from architect import Architect
import acc_profiler 
import latency_profiler

from pathlib import Path

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

filename = "gen_latencies"
CIFAR_CLASSES = 10
CIFAR_INPUT_BATCH = 1
CIFAR_INPUT_CHANNEL = 3
CIFAR_INPUT_SIZE = 32
NUM_SAMPLE = 300
CLUSTERS = 8

def generate_macro(dataset_name, 
                  test_loader,
                  seed, 
                  init_channels, 
                  layers, 
                  auxiliary, 
                  drop_path_prob, 
                  device):

  np.random.seed(seed)
  if torch.cuda.is_available():
    logging.info('gpu device available')
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
  else:
    logging.info('no gpu device available, use CPU')
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  criterion = nn.CrossEntropyLoss()
  macro_network = MacroNetwork(init_channels, CIFAR_CLASSES, 
    layers, criterion, device=device)
  micro_genotypes = pd.read_csv(filename + '_{}_center.csv'.format(str(device)))
  print(micro_genotypes) 
  sampled_architecture = macro_network.sample_architecture(dataset_name, NUM_SAMPLE, 
    micro_genotypes, init_channels, layers, auxiliary)
  model_df_with_acc_and_lat = profile_arch_lat_and_acc(dataset_name, 
    test_loader, sampled_architecture, criterion, device, drop_path_prob)
  model_df_with_acc_and_lat.to_csv(Path(filename + '_architecture_{}.csv'.format(str(device))), 
   index = None)

  return model_df_with_acc_and_lat

def profile_arch_lat_and_acc(dataset_name, test_loader, sampled_architectures, criterion, device, drop_path_prob):
  column = ['architecture',
            'mean_lat',
            'lat95',
            'lat99',
            'std_dev_lat',
            'macs',
            'params',
            'acc']
  df_with_lat = pd.DataFrame(columns=column)
  dict_list = []

  if dataset_name == "cifar10":
    input = torch.zeros(CIFAR_INPUT_BATCH, CIFAR_INPUT_CHANNEL, 
      CIFAR_INPUT_SIZE, CIFAR_INPUT_SIZE).to(device)
  else:
    sys.exit('Error!, dataset name not defined')
  
  for architecture in sampled_architectures:
    model = architecture["model"].to(device)
    model.drop_path_prob = drop_path_prob
    #profile parameters
    macs, params = profile(model, inputs=(input, ))
    #profile latencies
    mean_lat, latencies = latency_profiler.test_latency(model, input, device)
    #profile accuracy
    valid_acc, valid_obj = acc_profiler.infer(test_loader, model, criterion, device)

    dict_list.append({
        'architecture_name': architecture["name"],
        'mean_lat':mean_lat,
        'lat95':latencies[94],
        'lat99':latencies[98],
        'std_dev_lat': stdev(latencies),
        'macs':macs,
        'params':params,
        'acc': valid_acc
      })
  
    #print(architecture)
    print("============================")
  
  model_df_with_acc_and_lat = pd.DataFrame.from_dict(dict_list)
  
  return model_df_with_acc_and_lat

if __name__ == '__main__':
  seed = 0
  init_channels = 36
  layers = 12
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
    auxiliary,    
    drop_path_prob, 
    device) 
