import os
import sys
import glob

import logging

import torch
import torch.nn as nn
import torch.utils
from torch.autograd import Variable
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
from model_search import Network
from architect import Architect
import latency_profiler

from pathlib import Path
import argparse

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

filename = "generated_micro"
# We assume that model latencies are not affected by input data from dataset 
# Model latencies will be affected by batch size, input channel, and input size 
# This micro architecture profile doesn't need exact latencies number, 
# but it needs some comparison point between each genotypes. 
# As long as the comparison is consistent,
# the latencies comparison should be valid. 
# We use CIFAR because it's fast  
CIFAR_CLASSES = 10
INPUT_BATCH = 1
INPUT_CHANNEL = 3
INPUT_SIZE = 32
NUM_SAMPLE = 300
CLUSTERS = 8

def generate_micro(seed, gpuid, init_channels, layers, auxiliary, drop_path_prob, device):
  np.random.seed(seed)
  
  if torch.cuda.is_available():
    logging.info('gpu device available')
    torch.cuda.set_device(gpuid)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    logging.info('gpu device = %d' % gpuid)
  else:
    logging.info('no gpu device available, use CPU')

  # We assume that the micro architecture will be tested one a specific hardware of cpu
  # or a specific gpu that we want to test on. 
  # As long as the comparison is consistent, the latencies comparison should be valid.
  device = torch.device('cuda:'+ str(gpuid) if torch.cuda.is_available() else 'cpu')
  
  criterion = nn.CrossEntropyLoss()
  DARTS_network = Network(init_channels, CIFAR_CLASSES, 
    layers, criterion, device=device)
  
  sampled_genotypes = DARTS_network.sample_genotypes(NUM_SAMPLE)
  df_with_lat = profile_sample_latencies(sampled_genotypes, init_channels, layers, 
    auxiliary, drop_path_prob, device)
  #save after all data collected
  df_with_lat.to_csv(Path(filename + '_{}.csv'.format(str(device))), 
   index = None)

  df_with_lat = pd.read_csv(filename + '_{}.csv'.format(str(device))) 
  centers_genotype = kmedioid_grouping(df_with_lat, CLUSTERS)
  df_centers = pd.concat(centers_genotype) 
  df_centers.to_csv(Path(filename + '_{}_center.csv'.format(str(device))), 
    index = None)
  return df_centers

def profile_sample_latencies(sampled_genotypes, init_channels, layers, 
    auxiliary, drop_path_prob, device):
  column = ['genotype','mean_lat','lat95','lat99','std_dev_lat','macs','params']
  df_with_lat = pd.DataFrame(columns=column)
  dict_list = []
  for genotype in sampled_genotypes:
    # We assume that model latencies are not affected by input dataset type
    # The profile doesn't need exact latencies number, but needed as a comparison
    # point between each genotypes. As long as the comparison is consistent,
    # the latencies comparison should be valid. We use NetworkCIFAR because it's fast  
    model = NetworkCIFAR(init_channels, CIFAR_CLASSES, 
      layers, auxiliary, genotype)
    model = model.to(device)
    
    model.drop_path_prob = drop_path_prob
    input = torch.zeros(INPUT_BATCH, INPUT_CHANNEL, 
      INPUT_SIZE, INPUT_SIZE).to(device)
    macs, params = profile(model, inputs=(input, ))
    mean_lat, latencies = latency_profiler.test_latency(model, input, device)
    dict_list.append({
        'genotype': genotype,
        'mean_lat':mean_lat,
        'lat95':latencies[94],
        'lat99':latencies[98],
        'std_dev_lat': stdev(latencies),
        'macs':macs,
        'params':params
      })
    #print(genotype)
    print("============================")
  df_with_lat = pd.DataFrame.from_dict(dict_list)
  return df_with_lat

def kmedioid_grouping(df, nclusters):
  lat_data = df[['mean_lat', 'lat95', 'lat99']].copy()
  np_data = lat_data.to_numpy()

  kmedoids = KMedoids(metric="euclidean", n_clusters=nclusters, random_state=0).fit(np_data)
  label = kmedoids.labels_
  df['label'] = label
  #print(df)
  center = kmedoids.cluster_centers_
  #print(center)
  centers_genotype = []

  for i in range(nclusters):
    #print("center {} x:{}, y:{}, z:{}".format(i, center[i][0],center[i][1],center[i][2]))
    centers_genotype.append(df[
                      (df['mean_lat']==center[i][0]) & 
                      (df['lat95']==center[i][1]) & 
                      (df['lat99']==center[i][2])
    ])
  
  return centers_genotype

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--layers', type=int, default=25, help='number of layers')
  args = parser.parse_args()

  seed = 0
  gpuid = 7 
  init_channels = 36
  layers = args.layers
  auxiliary = False
  drop_path_prob = 0.2
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  list_genotypes = generate_micro(seed, gpuid, init_channels, layers, auxiliary, drop_path_prob, device) 
