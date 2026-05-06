import torch
import os
import subprocess
import sys
import json
from datetime import datetime

# =====================================================
# DEPENDENCY CHECK AND INSTALLATION
# =====================================================
def install_package(package):
    """Install a package if missing"""
    try:
        __import__(package)
        print(f"✅ {package} already installed")
        return True
    except ImportError:
        print(f"⚠️ Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "-q"])
        return True
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

# First, ensure accelerate is installed before any import
print("Checking dependencies...")
packages_to_install = ["accelerate", "transformers", "datasets", "peft", "bitsandbytes", "torch"]
for package in packages_to_install:
    install_package(package)

# Now import all packages
import accelerate
from accelerate import Accelerator
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

# =====================================================
# CONFIGURATION
# =====================================================
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "training/dataset/ediath_dataset.jsonl"
OUTPUT_DIR = "models/EDIATH-lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create logs directory
os.makedirs("logs", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔥 Using device: {device}")

if device == "cuda":
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

MAX_LENGTH = 512  # Increased for better context

# =====================================================
# TOKENIZER
# =====================================================
print("\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side for causal LM
tokenizer.padding_side = "right"

print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

# =====================================================
# MODEL SETUP (FIXED BY REMOVING DEVICE_MAP)
# =====================================================
if device == "cuda":
    print("\n🚀 Setting up GPU mode with 4-bit quantization (RTX 3050)")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load model WITHOUT device_map (let it auto-detect)
    # This avoids the accelerate check error
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache=False  # Disable cache for gradient checkpointing
    )
    
    # Move model to GPU explicitly
    model = model.to(device)

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    use_fp16 = True
    optimizer = "paged_adamw_8bit"
    
    print("✅ Model loaded with 4-bit quantization")

else:
    print("\n⚠️ Loading CPU model (no quantization)")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model = model.to("cpu")
    use_fp16 = False
    optimizer = "adamw_torch"
    
    print("✅ Model loaded on CPU")

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Total model parameters: {total_params:,}")

# =====================================================
# LORA CONFIGURATION
# =====================================================
print("\n🔧 Configuring LoRA...")

lora_config = LoraConfig(
    r=8,  # Reduced from 16 to save memory
    lora_alpha=16,
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
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ LoRA applied. Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# =====================================================
# SAFE LOAD DATASET
# =====================================================
def load_jsonl_safe(path):
    """Safely load JSONL dataset with error handling"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    clean_data = []
    bad_lines = 0

    print(f"\n📂 Loading dataset from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                if "messages" not in obj:
                    raise ValueError("Missing 'messages' key")
                
                if not isinstance(obj["messages"], list):
                    raise ValueError("'messages' must be a list")
                
                if len(obj["messages"]) == 0:
                    raise ValueError("'messages' list is empty")

                clean_data.append(obj)

            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"⚠️ Line {i}: JSON decode error - {e}")
            except Exception as e:
                bad_lines += 1
                print(f"⚠️ Line {i}: {e}")

    print(f"✅ Loaded {len(clean_data)} valid samples")
    if bad_lines > 0:
        print(f"⚠️ Skipped {bad_lines} invalid samples")
    
    if len(clean_data) == 0:
        raise ValueError("No valid samples found in dataset")

    return clean_data

# Load dataset
raw_data = load_jsonl_safe(DATASET_PATH)

# Convert to dataset
dataset = Dataset.from_list(raw_data)

# Split into train/test (smaller test set to save memory)
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"📊 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# =====================================================
# TOKENIZATION FUNCTION (SIMPLE VERSION)
# =====================================================
def tokenize_function(examples):
    """Simple tokenization function"""
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]
        
        # Extract user and assistant messages
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]
        
        # Format for Mistral
        prompt = f"<s>[INST] {user_content} [/INST]"
        
        # Full text
        full_text = f"{prompt} {assistant_content} {tokenizer.eos_token}"
        
        # Tokenize
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
        
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # Mask the prompt tokens (set labels to -100 for prompt part)
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)
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

# Apply tokenization
print("\n🔄 Tokenizing dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1,  # Process one at a time to avoid memory issues
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training set"
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing evaluation set"
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
    
    # Batch settings (optimized for 6GB GPU)
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Reduced from 8 to save memory
    
    # Training settings
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim=optimizer,
    max_grad_norm=1.0,
    
    # Evaluation (reduce frequency to save time)
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Mixed precision
    fp16=use_fp16,
    bf16=False,
    
    # Logging
    logging_steps=10,
    logging_dir="logs",
    report_to="none",
    
    # Performance
    dataloader_num_workers=0,  # Windows compatibility
    group_by_length=False,  # Memory optimization
    
    # Memory
    gradient_checkpointing=True if device == "cuda" else False,
    
    # Remove unused columns
    remove_unused_columns=False,
    
    # Seed
    seed=42,
    
    # Debug
    ignore_data_skip=True
)

# Calculate effective batch size
effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
print(f"📈 Effective batch size: {effective_batch}")
print(f"📉 Learning rate: {training_args.learning_rate}")
print(f"🔄 Training epochs: {training_args.num_train_epochs}")

# =====================================================
# TRAINER SETUP
# =====================================================
print("\n🏋️ Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# =====================================================
# TRAINING
# =====================================================
print("\n🚀 Starting training...")
print("="*60)

try:
    # Start training
    train_result = trainer.train()
    
    # Save training metrics
    with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
        json.dump({
            "train_loss": float(train_result.training_loss),
            "global_step": train_result.global_step,
            "epoch": float(train_result.epoch),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Final training loss: {train_result.training_loss:.4f}")
    print(f"📈 Global steps: {train_result.global_step}")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =====================================================
# SAVE MODEL
# =====================================================
print("\n💾 Saving model...")

# Save LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training configuration
trainer.save_state()

# Save final training info
config_info = {
    "base_model": BASE_MODEL,
    "lora_config": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config.target_modules
    },
    "training_params": {
        "max_length": MAX_LENGTH,
        "num_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation": training_args.gradient_accumulation_steps,
        "effective_batch": effective_batch,
        "use_fp16": use_fp16
    },
    "final_train_loss": float(train_result.training_loss),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config_info, f, indent=2)

print(f"✅ Model saved to: {OUTPUT_DIR}")

# =====================================================
# CLEANUP
# =====================================================
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("🎉 EDIATH LoRA training complete!")
print("="*60)
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"🔧 Next steps:")
print(f"   1. Test the model: python test_lora.py")
print(f"   2. Merge with base model: python merge_lora.py")
print(f"   3. Convert to GGUF: python convert_to_gguf.py")
print("="*60)
