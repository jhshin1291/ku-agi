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
import evaluate

prefix = "summarize: "

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

## evaluation of DialogSum dataset using rouge score

module_list = ['/home/work/data_yhgo/ku-agi/evaluation/checkpoint/fine-tune_ccdv-sum_epoch01/', '/home/work/data_yhgo/ku-agi/evaluation/checkpoint//fine-tune_GEM-xlsum_epoch0.07/']
merged_dir = '/home/work/data_yhgo/ku-agi/evaluation/checkpoint/'
peft_model_id = "./checkpoint/0to10/"
config = PeftConfig.from_pretrained(peft_model_id)


# base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "right"  # Fix for fp16

tokenizer = base_tokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    # base_model_name,
    config.base_model_name_or_path,
    # device_map={"": 0}
    device_map="auto"
)

rouge = evaluate.load('rouge')
# test_data = ld("billsum", split="ca_test")
test_data = ld("billsum", split="ca_test[10:13]")

# [5] Get max length of input data 
lenmax = 0
for item in test_data:
  lenitem = len(item['text'])
  if lenitem > lenmax:
    lenmax = lenitem

for module in module_list:
    fine_tuned_model = PeftModel.from_pretrained(base_model, module, local_files_only=True)
    
    #TODO: evaluate fine-tuned model respectively: 1) paper summarization, 2) article summarization
    
    # res = fine_tuned_model(tokenized_billsum[0]['input_ids'])
    # [7] Evaluating rouge-score
    references = []
    predictions = []

    fine_tuned_model = pipeline(task="summarization", model=fine_tuned_model, tokenizer=base_tokenizer, max_length=4000)

    for item in test_data:
        # print("item", item)
        reference = item['summary']
        references.append(reference)

        prediction = fine_tuned_model.predict(item['text'])
        predictions.append(prediction[0]['summary_text'])

        
    results = rouge.compute(predictions=predictions,
                            references=references)

    print(module, results)


for x in range(0, 11): 
    i = x
    j = 10 - x
    references = []
    predictions = []
    merged_model = PeftModel.from_pretrained(base_model, merged_dir + str(i) + 'to' + str(j) + '/', local_files_only=True)
    #TODO: evaluate merged model
    merged_model = pipeline(task="summarization", model=merged_model, tokenizer=base_tokenizer, max_length=4096)

    for item in test_data:
        reference = item['summary']
        references.append(reference)

        prediction = merged_model.predict(item['text'])
        predictions.append(prediction[0]['summary_text'])

        
    results = rouge.compute(predictions=predictions,
                            references=references)

    print("merged proportion to", i, "", j, results)
