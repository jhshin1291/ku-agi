#!/usr/bin/python

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from trl import SFTTrainer
import re
import pdb

import evaluate

# [1] Setting Dataset & Basemodel
# Dataset
# data_name = "mlabonne/guanaco-llama2-1k"
data_name = "ccdv/arxiv-summarization"
test_data = load_dataset(data_name, split="test")

# Model and tokenizer names
# base_model_name = "NousResearch/Llama-2-7b-chat-hf"
base_model_name = "NousResearch/Llama-2-7b-hf"
# fine_tuned_model_name = "oslab/llama-2-7b-oslab2"
fine_tuned_model_name = "/home/work/data_yhgo/cyshin/agi/fine-tune_ccdv-sum_epoch01/"

# [2] Creating Llama2 Tokenizer
# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# [3] Creating configuration for Quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# [4] Instantiating base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto" # accelerate library
    # device_map={"": 0}

)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# [5] Get max length of input data 
lenmax = 0
for item in test_data:
  lenitem = len(item['article'])
  if lenitem > lenmax:
    lenmax = lenitem

# [6] Loading fine-tuned model
lora_config = LoraConfig.from_pretrained(fine_tuned_model_name)
fine_tuned_model = get_peft_model(base_model, lora_config)

base_model = pipeline(task="summarization", model=base_model_name, tokenizer=llama_tokenizer, max_length=lenmax)
fine_tuned_model = pipeline(task="summarization", model=fine_tuned_model, tokenizer=llama_tokenizer, max_length=lenmax)

# [7] Evaluating rouge-score
references = []
predictions = []

rouge = evaluate.load("rouge")
for item in test_data:
  reference = item['abstract']
  references.append(reference)

  prediction = fine_tuned_model.predict(item['article'])
  predictions.append(prediction[0]['summary_text'])
  
results = rouge.compute(predictions=predictions,
                        references=references)

print(results)

# [7] Evaluating rouge-score with evaluator
# task_evaluator = evaluate.evaluator("summarization")
# results = task_evaluator.compute(model_or_pipe=fine_tuned_model, data=test_data, metric=rouge, input_column="article", output_column="abstract")
# print(results)






