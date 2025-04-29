# ====================== 🔧 SETUP + UPLOAD ======================
!pip install -q bitsandbytes accelerate peft transformers datasets

from pathlib import Path
import os
import torch
import zipfile
import shutil

# Set CUDA options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 🔹 Create folders
Path("model").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

from google.colab import drive
drive.mount('/content/drive')

# File paths
model_zip_path = "HERE_GOES_PATH_TO_MODEL_ZIP"  # Replace with the path to the model ZIP file
dataset_path = "HERE_GOES_PATH_TO_DATASET"  # Replace with the path to the dataset file

# Check if the model ZIP file exists
if os.path.exists(model_zip_path):
    try:
        # Unzip the file into the 'model' folder
        if model_zip_path.endswith(".zip"):
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall("model")  # Unzip the model into the "model" folder
            print("The model has been successfully unpacked.")
    except Exception as e:
        print(f"Failed to unpack the ZIP file: {e}")
else:
    print(f"The model ZIP file does not exist at the path: {model_zip_path}")

# Check if the dataset exists
if os.path.exists(dataset_path):
    try:
        # Copy the dataset to the 'data' folder
        shutil.copy(dataset_path, "data/YOUR DATASET NAME")  # Copy the dataset to the "data" folder
        print("The dataset has been successfully copied.")
    except Exception as e:
        print(f"Failed to copy the dataset: {e}")
else:
    print(f"The dataset file does not exist at the path: {dataset_path}")

# ====================== 🔍 FINE-TUNE ======================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def fine_tune_with_qlora():
    model_name_or_path = "model"  # already unpacked model
    dataset_path = "data/YOUR DATASET NAME"
    output_dir = "HERE_GOES_OUTPUT_DIR"  # Replace with the desired output directory

    # 🔹 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🔹 BitsAndBytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 🔹 Model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model = prepare_model_for_kbit_training(model)

    # 🔹 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # 🔹 Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        prompt = (example["instruction"] + "\n" + example["input"]).strip()
        output = example["output"].strip()

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        output_ids = tokenizer(output, add_special_tokens=False).input_ids

        # Concatenated input_ids
        input_ids = prompt_ids + output_ids

        # Truncate or pad to max 512 tokens
        if len(input_ids) > 512:
            input_ids = input_ids[:512]
    
        attention_mask = [1] * len(input_ids)

        # Labels: -100 for the prompt (to ignore it), normal tokens for the output
        labels = [-100] * len(prompt_ids) + output_ids
        if len(labels) > 512:
            labels = labels[:512]

        # Pad to ensure equal lengths
        pad_len = 512 - len(input_ids)

        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_dataset = dataset.map(tokenize)

    # 🔹 Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        max_steps=800,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    print("✅ Starting fine-tuning")
    trainer.train()
    print("✅ Saving the model with LoRA")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 🔁 Start training
fine_tune_with_qlora()
