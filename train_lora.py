from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import json
import torch
from peft import LoraConfig, TaskType, get_peft_model

from swanlab.integration.transformers import SwanLabCallback

def process_func(example):
    MAX_LENGTH = 4096    
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    with open('./filter_length_data.jsonl', 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    df = pd.DataFrame(data)
    ds = Dataset.from_pandas(df)

    model_path = '/home/user/storage/szx/model/Qwen2___5-3B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    swanlab_callback = SwanLabCallback(
        project="Qwen-Distill",
        experiment_name="filter_length_data",
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )

    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir="./output/filter_length_data",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type='cosine',
        logging_steps=10,
        num_train_epochs=3,
        warmup_ratio=0.3,
        bf16=True,
        save_steps=100, 
        learning_rate=5e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback]
    )
    trainer.train()
