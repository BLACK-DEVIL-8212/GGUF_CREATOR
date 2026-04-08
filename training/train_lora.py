import torch
import os
import subprocess
import sys
import json
try:
    import accelerate
except ImportError:
    print("⚠️ Installing accelerate...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
    import accelerate
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "training/dataset/ediath_dataset.jsonl"
OUTPUT_DIR = "models/EDIATH-lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

if device == "cuda":
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")

MAX_LENGTH = 256

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# MODEL SETUP (STRICT GPU MODE)
# =====================================================

if device == "cuda":

    print("🚀 FORCE GPU MODE (RTX 3050)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 🔥 NO TRY/EXCEPT → FAIL IF GPU FAILS (IMPORTANT)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",  # FORCE GPU
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    use_fp16 = True
    optimizer = "paged_adamw_8bit"

else:

    print("⚠️ Loading CPU model")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        trust_remote_code=True
    )

    model.to("cpu")

    use_fp16 = False
    optimizer = "adamw_torch"

# =====================================================
# LORA CONFIG
# =====================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================================================
# SAFE LOAD DATASET
# =====================================================

def load_jsonl_safe(path):
    clean_data = []
    bad_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                if "messages" not in obj:
                    raise ValueError("Missing 'messages' key")

                clean_data.append(obj)

            except Exception as e:
                bad_lines += 1
                print(f"⚠️ Skipping bad JSON line {i}: {e}")

    print(f"✅ Loaded {len(clean_data)} valid samples")
    print(f"❌ Skipped {bad_lines} bad samples")

    return clean_data


raw_data = load_jsonl_safe(DATASET_PATH)

dataset = Dataset.from_list(raw_data)
dataset = dataset.train_test_split(test_size=0.05)

# =====================================================
# TOKENIZATION
# =====================================================

def tokenize(example):

    system = ""
    user = ""
    assistant = ""

    for msg in example["messages"]:

        if msg["role"] == "system":
            system = msg["content"]

        elif msg["role"] == "user":
            user = msg["content"]

        elif msg["role"] == "assistant":
            assistant = msg["content"]

    prompt = f"""<|system|>
{system}

<|user|>
{user}

<|assistant|>
"""

    full_text = prompt + assistant + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )["input_ids"]

    labels = input_ids.copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

dataset["train"] = dataset["train"].map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

dataset["test"] = dataset["test"].map(
    tokenize,
    remove_columns=dataset["test"].column_names
)

# =====================================================
# DATA COLLATOR
# =====================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False
)

# =====================================================
# TRAINING CONFIG (FIXED)
# =====================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16 if device == "cuda" else 4,

    num_train_epochs=3,

    learning_rate=1e-4,

    weight_decay=0.01,

    max_grad_norm=1.0,

    logging_steps=10,

    save_steps=200,

    save_total_limit=2,

    fp16=use_fp16,

    optim=optimizer,

    warmup_steps=50,  # ✅ FIXED

    lr_scheduler_type="cosine",

    report_to="none"
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"],

    data_collator=data_collator
)

# =====================================================
# TRAIN
# =====================================================

trainer.train()

# =====================================================
# SAVE MODEL
# =====================================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ EDIATH LoRA training complete:", OUTPUT_DIR)