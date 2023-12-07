#!/usr/bin/python

import torch
from torch import Tensor
from pprint import pprint as pp
import os.path
import pdb


f_adapter1 = 'oslab/llama-2-7b-oslab/adapter_model.bin'
f_adapter2 = 'oslab_with_chat/llama-2-7b-oslab/adapter_model.bin'

if not os.path.exists(f_adapter1):
    raise Exception(f"file not found exception -> {f_adapter1}")

if not os.path.exists(f_adapter1):
    raise Exception(f"file not found exception -> {f_adapter2}")

adapter_w1 = torch.load(f_adapter1, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
adapter_w2 = torch.load(f_adapter2, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for k, v in adapter_w1.items():
    if not isinstance(v, Tensor):
        raise Exception("adapter value is not a Tensor type")
    adapter_w1[k] = adapter_w1[k] + adapter_w2[k]

torch.save(adapter_w1, 'oslab/new/new_adapter_model.bin')

print("over")
