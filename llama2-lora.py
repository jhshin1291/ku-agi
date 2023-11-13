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

# [1] Setting Dataset & Basemodel
# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
fine_tuned_model_name = "oslab/llama-2-7b-oslab"

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
    task_type="CAUSAL_LM"
)


# [6] Creating training parameters
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
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
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)


# [8] do fine-tuning..
fine_tuning.train()

# [9] save the fine-tuned model object
fine_tuning.model.save_pretrained(fine_tuned_model_name)

# [10] gen text by fine-tuned model
# Generate Text
lora_config = LoraConfig.from_pretrained(fine_tuned_model_name)
fine_tuned_model = get_peft_model(base_model, lora_config)
#==> fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_name)

llama2_chat_model = pipeline(task="text-generation", model=base_model_name, tokenizer=llama_tokenizer, max_length=200, device=1)
fine_tuned_model = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=llama_tokenizer, max_length=200, device=0)


def chat_with(model_name, query):
    global fine_tuned_model
    global llama2_chat_model

    model_dict = {
        "llama2-chat": llama2_chat_model,
        "llama2-chat-oslab": fine_tuned_model
    }
    model = model_dict[model_name]

    maxlen = max([len(model_name) for model_name in model_dict.keys()])
    model_name = model_name.ljust(maxlen)
    print(f"[{model_name}] generating the answer ....")
    output = model(f"<s>[INST] {query} [/INST]")
    output = re.sub("^.*\[\/INST\]  ", "", output[0]['generated_text'])
    print(f"[{model_name}] {output}\n\n")


while True:
    print("=" * 60)
    query = input(f"Hi, sir. What can I do for you? >> ")
    if query.lower() == "quit":
        print("Program is over. bye")
        exit(0)

    chat_with("llama2-chat", query)
    chat_with("llama2-chat-oslab", query)


