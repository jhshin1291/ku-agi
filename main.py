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
from trl import SFTTrainer

import pdb 

# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
#refined_model = "llama-2-7b-mlabonne-enhanced" #You can give it your own name
#refined_model = "HuggingFaceH4/zephyr-7b-alpha"
refined_model = "mlabonne/llama-2-7b-miniguanaco"

#base_model_name = "NousResearch/Llama-2-13b-chat-hf"
#refined_model = "llama-2-13b-mlabonne-enhanced" #You can give it your own name

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)


# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1 

# LoRA Config (LongLoRA, QLoRA)
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
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


# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)


# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)

# Generate Text
text_gen = pipeline(task="text-generation", model=refined_model, tokenizer=llama_tokenizer, max_length=200, device=1)
while True:
        #query = "How do I use the OpenAI API?"
        oslab_bot = "Hi, sir. What can I do for you? >> "
        query = input("[OSlab-bot] " + oslab_bot)
        print(f"[OSlab-bot] your question is -> {query}")
        print("[OSlab-bot] generating the answer ....")
        output = text_gen(f"<s>[INST] {query} [/INST]")
        print("[OSlab-bot] " + output[0]['generated_text'])
        print("")
