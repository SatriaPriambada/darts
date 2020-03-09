import time

import torch
import torch.nn as nn

def test_latency(model, dummy_input, device):
  n_warmup = 20
  n_sample = 100
  measured_latency = {'warmup': [], 'sample': []}

  model.eval()
  with torch.no_grad():
    for i in range(n_warmup + n_sample):
      start_time = time.time()
      model(dummy_input)
      used_time = (time.time() - start_time) * 1e3  # ms
      if i >= n_warmup:
        measured_latency['sample'].append(used_time)
      else:
        measured_latency['warmup'].append(used_time)

  return (sum(measured_latency['sample']) / n_sample), sorted(measured_latency['sample'])
