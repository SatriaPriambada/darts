import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from model_search import Network
from model_search import MixedOp
from model_search import Cell
from model import HeterogenousNetworkImageNet
from model import HeterogenousNetworkCIFAR
import random
import itertools

class MacroNetwork(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, device='cuda'):
    super(MacroNetwork, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas(device)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self, device):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    if device == "cuda":
      self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    else:
      self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def sample_architecture(self, dataset_name, nsample, micro_genotypes, init_channels, 
    layers, auxiliary):
    architectures = []
    model_names = [] 
    valid_gen_choice = micro_genotypes['genotype'].tolist()
    valid_gen_choice.append("none")
    for _ in range(nsample):
      selected_layers = []
      for i in range(layers):
        selected_layers.append(random.choice(valid_gen_choice))
      name = ';'.join([str(elem) for elem in selected_layers]) 
      if dataset_name == "cifar10":
        CIFAR_CLASSES = 10
        architectures.append({
          "name": name,
          "model": HeterogenousNetworkCIFAR(
            init_channels, 
            CIFAR_CLASSES, 
            layers, 
            auxiliary, 
            selected_layers
          )
        }) 
      elif dataset_name == "imagenet":
        IMAGENET_CLASSES = 1000
        architectures.append({
          "name": name,
          "model": HeterogenousNetworkImageNet(
            init_channels, 
            IMAGENET_CLASSES, 
            layers, 
            auxiliary, 
            selected_layers
          )
        }) 
    return architectures

