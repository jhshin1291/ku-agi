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

DO_FINE_TUNING = False

# [1] Setting Dataset & Basemodel
# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
#base_model_name = "NousResearch/Llama-2-7b-hf"
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
fine_tuned_model_name = "oslab/llama-2-7b-oslab2"

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
    #device_map="auto"
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
    # num_train_epochs=1,
    num_train_epochs=0.001,
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
    args=train_params,
    max_seq_length=1024
)


# [8] do fine-tuning..
# fine_tuning -> <trl.trainer.sft_trainer.SFTTrainer object at 0x7f5aec29fc70>
# train function -> /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(1460)train()
if DO_FINE_TUNING:
    fine_tuning.train()

# [9] save the fine-tuned model object
# fine_tuning.model -> <class 'peft.peft_model.PeftModelForCausalLM'>
# save_pretrained function -> 
# during training, save_pretrained is called to save check-point
'''
(Pdb) w
  /home/work/data_yhgo/sjh/llama2-LoRA/llama2-lora.py(97)<module>()
-> fine_tuning.train()
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(1483)train()
-> if resume_from_checkpoint is False:
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(1901)_inner_training_loop()
-> self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(2237)_maybe_log_save_evaluate()
-> self._save_checkpoint(model, trial, metrics=metrics)
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(2294)_save_checkpoint()
-> self.save_model(output_dir, _internal_call=True)
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(2769)save_model()
-> self._save(output_dir)
  /home/work/.local/lib/python3.9/site-packages/transformers/trainer.py(2827)_save()
-> self.model.save_pretrained(
> /home/work/.local/lib/python3.9/site-packages/peft/peft_model.py(144)save_pretrained()
-> if os.path.isfile(save_directory):
'''
if DO_FINE_TUNING:
    fine_tuning.model.save_pretrained(fine_tuned_model_name)

'''
import gc
torch.cuda.empty_cache()
gc.collect()
del base_model
del fine_tuning
del llama_tokenizer
'''

# [10] gen text by fine-tuned model
# Generate Text
# from_pretrained -> <bound method PeftConfigMixin.from_pretrained of <class 'peft.tuners.lora.LoraConfig'>>
'''
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
'''
lora_config = LoraConfig.from_pretrained(fine_tuned_model_name)
fine_tuned_model = get_peft_model(base_model, lora_config)
# == fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_name)

llama2_chat_model = pipeline(task="text-generation", model=base_model_name, tokenizer=llama_tokenizer, max_length=200, device=1)
ft_model          = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=llama_tokenizer, max_length=200, device=0)

def chat_with(model_name, query):
    global ft_model
    global llama2_chat_model

    model_dict = {
        "llama2-chat": llama2_chat_model,
        "llama2-chat-oslab": ft_model
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


