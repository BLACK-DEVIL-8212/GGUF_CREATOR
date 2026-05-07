import os
import sys
import json
import torch
import subprocess

# =====================================================
# INSTALL PACKAGES
# =====================================================

def install_package(package):

    try:
        __import__(package)
        print(f"✅ {package} already installed")

    except ImportError:

        print(f"⚠️ Installing {package}...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package,
            "--upgrade",
            "-q"
        ])

print("🔍 Checking dependencies...")

packages = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "bitsandbytes",
    "sentencepiece"
]

for package in packages:
    install_package(package)

# =====================================================
# IMPORTS
# =====================================================

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# =====================================================
# CONFIG
# =====================================================

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

DATASET_PATH = "training/dataset/ediath_dataset.jsonl"

OUTPUT_DIR = "models/EDIATH-lora"

MAX_LENGTH = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# GPU SETUP
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🔥 Device: {device}")

if device == "cuda":

    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"💾 GPU Memory: {gpu_mem:.2f} GB")

    torch.cuda.empty_cache()

# =====================================================
# TOKENIZER
# =====================================================

print("\n📝 Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("✅ Tokenizer loaded")

# =====================================================
# QUANTIZATION CONFIG
# =====================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\n🚀 Loading model...")

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True,

    torch_dtype=torch.float16,

    use_cache=False
)

print("✅ Model loaded")

# =====================================================
# PREPARE MODEL
# =====================================================

model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable()

# =====================================================
# LORA CONFIG
# =====================================================

print("\n🔧 Applying LoRA...")

lora_config = LoraConfig(

    r=4,

    lora_alpha=8,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# =====================================================
# LOAD DATASET
# =====================================================

def load_jsonl_safe(path):

    clean_data = []

    bad_lines = 0

    print(f"\n📂 Loading dataset: {path}")

    with open(path, "r", encoding="utf-8") as f:

        lines = f.readlines()

    for i, line in enumerate(lines):

        try:

            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "messages" not in obj:
                continue

            clean_data.append(obj)

        except Exception as e:

            bad_lines += 1

            if bad_lines <= 10:
                print(f"⚠️ Line {i}: {e}")

    print(f"\n✅ Valid samples: {len(clean_data)}")

    print(f"⚠️ Invalid skipped: {bad_lines}")

    return clean_data

raw_data = load_jsonl_safe(DATASET_PATH)

dataset = Dataset.from_list(raw_data)

split_dataset = dataset.train_test_split(
    test_size=0.05,
    seed=42
)

train_dataset = split_dataset["train"]

eval_dataset = split_dataset["test"]

print(f"\n📊 Train samples: {len(train_dataset)}")

print(f"📊 Eval samples: {len(eval_dataset)}")

# =====================================================
# TOKENIZATION
# =====================================================

def tokenize_function(examples):

    input_ids_list = []

    attention_mask_list = []

    labels_list = []

    for messages in examples["messages"]:

        user_content = ""

        assistant_content = ""

        for msg in messages:

            if msg["role"] == "user":
                user_content = msg["content"]

            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        prompt = f"<s>[INST] {user_content} [/INST]"

        full_text = (
            f"{prompt} "
            f"{assistant_content} "
            f"{tokenizer.eos_token}"
        )

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

        input_ids = tokenized["input_ids"]

        attention_mask = tokenized["attention_mask"]

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

        prompt_len = len(prompt_tokens["input_ids"])

        labels = input_ids.copy()

        labels[:prompt_len] = [-100] * prompt_len

        input_ids_list.append(input_ids)

        attention_mask_list.append(attention_mask)

        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# =====================================================
# TOKENIZE DATASET
# =====================================================

print("\n🔄 Tokenizing dataset...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print("✅ Tokenization complete")

# =====================================================
# DATA COLLATOR
# =====================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

print("\n⚙️ Configuring training...")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-4,

    weight_decay=0.01,

    logging_steps=10,

    save_steps=500,

    save_total_limit=2,

    fp16=True,

    gradient_checkpointing=True,

    remove_unused_columns=False,

    dataloader_num_workers=0,

    report_to="none"
)

print("✅ Training arguments configured")

# =====================================================
# TRAINER
# =====================================================

print("\n🏋️ Initializing trainer...")

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_eval,

    data_collator=data_collator
)

print("✅ Trainer initialized")

# =====================================================
# TRAINING
# =====================================================

print("\n🚀 Starting training...")

print("=" * 60)

try:

    trainer.train()

    print("\n✅ Training completed!")

except Exception as e:

    print("\n❌ Training failed")

    print(str(e))

    import traceback

    traceback.print_exc()

    sys.exit(1)

# =====================================================
# SAVE MODEL
# =====================================================

print("\n💾 Saving model...")

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to: {OUTPUT_DIR}")

# =====================================================
# CLEANUP
# =====================================================

del model

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n🎉 TRAINING COMPLETE")
