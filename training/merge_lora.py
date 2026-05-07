import os
import shutil
import gc
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel

# =========================================================
# CONFIG
# =========================================================

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

LORA_PATH = "models/EDIATH-lora"

OUTPUT_PATH = "models/EDIATH-merged"

# =========================================================
# CLEAN OUTPUT FOLDER
# =========================================================

if os.path.exists(OUTPUT_PATH):

    print("⚠️ Removing old merged model...")
    shutil.rmtree(OUTPUT_PATH)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================

device = "cpu"

dtype = torch.float16

print("🚀 LOW RAM MERGE MODE")
print("=" * 60)

# =========================================================
# VALIDATION
# =========================================================

if not os.path.exists(LORA_PATH):

    raise FileNotFoundError(
        f"❌ LoRA folder not found: {LORA_PATH}"
    )

adapter_file = os.path.join(
    LORA_PATH,
    "adapter_config.json"
)

if not os.path.exists(adapter_file):

    raise FileNotFoundError(
        "❌ adapter_config.json missing.\n"
        "Your LoRA training may have failed."
    )

# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

torch.set_grad_enabled(False)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =========================================================
# LOAD BASE MODEL
# =========================================================

print("📥 Loading base model...")

try:

    base_model = AutoModelForCausalLM.from_pretrained(

        BASE_MODEL,

        torch_dtype=dtype,

        low_cpu_mem_usage=True,

        device_map="cpu",

        trust_remote_code=True
    )

    print("✅ Base model loaded")

except Exception as e:

    print(f"❌ Failed loading base model:\n{e}")
    exit(1)

# =========================================================
# LOAD TOKENIZER
# =========================================================

print("📥 Loading tokenizer...")

try:

    tokenizer = AutoTokenizer.from_pretrained(

        BASE_MODEL,

        use_fast=False,   # IMPORTANT FIX

        trust_remote_code=True
    )

    # optional safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Tokenizer loaded")

except Exception as e:

    print(f"❌ Tokenizer load failed:\n{e}")
    print("\n⚠️ FIX:")
    print("Run:")
    print("pip install -U transformers tokenizers sentencepiece")
    exit(1)

# =========================================================
# LOAD LORA
# =========================================================

print("📥 Loading LoRA adapter...")

try:

    model = PeftModel.from_pretrained(

        base_model,

        LORA_PATH,

        device_map="cpu",

        torch_dtype=dtype
    )

    print("✅ LoRA adapter loaded")

except Exception as e:

    print(f"❌ Failed loading LoRA:\n{e}")
    exit(1)

# =========================================================
# MERGE
# =========================================================

print("🔀 Merging LoRA into base model...")

try:

    merged_model = model.merge_and_unload()

    print("✅ Merge successful")

except Exception as e:

    print(f"❌ Merge failed:\n{e}")
    exit(1)

# =========================================================
# SAVE MODEL
# =========================================================

print("💾 Saving merged model...")

try:

    merged_model.save_pretrained(

        OUTPUT_PATH,

        safe_serialization=True,

        max_shard_size="2GB"
    )

    print("✅ Merged model saved")

except Exception as e:

    print(f"❌ Save failed:\n{e}")
    exit(1)

# =========================================================
# SAVE TOKENIZER
# =========================================================

print("💾 Saving tokenizer...")

try:

    tokenizer.save_pretrained(OUTPUT_PATH)

    print("✅ Tokenizer saved")

except Exception as e:

    print(f"❌ Tokenizer save failed:\n{e}")
    exit(1)

# =========================================================
# VERIFY OUTPUT
# =========================================================

print("🔎 Verifying saved files...")

required_files = [
    "config.json",
    "tokenizer_config.json",
]

missing_files = []

for file in required_files:

    path = os.path.join(OUTPUT_PATH, file)

    if not os.path.exists(path):
        missing_files.append(file)

if missing_files:

    print("❌ Missing files:")

    for f in missing_files:
        print(f"   - {f}")

    exit(1)

print("✅ Verification passed")

# =========================================================
# CLEANUP
# =========================================================

del model
del merged_model
del base_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =========================================================
# DONE
# =========================================================

print("=" * 60)

print("🎉 MERGE COMPLETED SUCCESSFULLY")

print(f"📁 Output Path: {OUTPUT_PATH}")

print("=" * 60)

print("📌 NEXT STEP")
print("Convert merged model to GGUF using llama.cpp")

print("=" * 60)
