#!/usr/bin/python

import torch
from torch import Tensor
from pprint import pprint as pp
import os.path
import pdb

import torch
import torch
from datasets import load_dataset as ld
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from transformers import default_data_collator

from peft import LoraConfig
from peft import PeftModel, PeftConfig
from peft import get_peft_model
from trl import SFTTrainer
import re
import pdb
from tqdm import tqdm

import copy
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from datasets import Dataset
from functools import partial
import pandas as pd
import nevergrad as ng
from torch.utils.data import DataLoader

f_adapter1 = './fine-tune_ccdv-sum_epoch01/adapter_model.bin'
f_adapter2 = './fine-tune_GEM-xlsum_epoch0.07/adapter_model.bin'

if not os.path.exists(f_adapter1):
    raise Exception(f"file not found exception -> {f_adapter1}")

if not os.path.exists(f_adapter1):
    raise Exception(f"file not found exception -> {f_adapter2}")

adapter_w1 = torch.load(f_adapter1, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
adapter_w2 = torch.load(f_adapter2, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

cnt = 0
for x in range(0, 11): 
    new_adapter = copy.deepcopy(adapter_w1)
    i = x/10
    j = (10 - x)/10

    print(i, j)
    for k, v in adapter_w1.items():
        if not isinstance(v, Tensor):
            raise Exception("adapter value is not a Tensor type")
        new_adapter[k] = i * adapter_w1[k] + j * adapter_w2[k]

    torch.save(new_adapter, '/home/work/data_yhgo/yhgo/' + str(x) + 'to' + str(10-x) + '/adapter_model.bin')

print("over")
