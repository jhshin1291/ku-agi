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

# settings
epochs = 0.07

DO_FINE_TUNING = False

# [1] Setting Dataset & Basemodel
# Dataset
# data_name = "mlabonne/guanaco-llama2-1k"
data_name = "GEM/xlsum"
#training_data = load_dataset(data_name, split="train")
dataset = load_dataset(data_name, "english")
training_data = dataset['train']
test_data  = dataset['test']
validation_data = dataset['validation']

# Model and tokenizer names
#base_model_name = "NousResearch/Llama-2-7b-chat-hf"
base_model_name = "NousResearch/Llama-2-7b-hf"
fine_tuned_model_name = "oslab/llama-2-7b-xlsum"

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
    device_map={"": 0}
    #device_map="auto" # accelerate library

)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1


# [5] Creating LoRA config based on PEFT parameters
# LoRA Config (LongLoRA, QLoRA)
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM")


# [6] Creating training parameters
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)


# [7] Creating Supervised Fine-Tuning trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    max_seq_length=1024,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# [10] gen text by fine-tuned model
# Generate Text
# from_pretrained -> <bound method PeftConfigMixin.from_pretrained of <class 'peft.tuners.lora.LoraConfig'>>
lora_config = LoraConfig.from_pretrained(fine_tuned_model_name)
fine_tuned_model = get_peft_model(base_model, lora_config)
#fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_name)

# do summarization task
# [1] create a summarization pipeline
summ_model = pipeline(task="summarization", model=fine_tuned_model, tokenizer=llama_tokenizer, max_length=2000, device=0)

# [2] predict(=summarization result)
ret_summ = summ_model.predict(test_data['text'][0])
print(ret_summ[0]['summary_text'])
