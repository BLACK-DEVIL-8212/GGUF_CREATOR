import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================================================
# 🔥 CONFIG
# =========================================================
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "models/EDIATH-lora"
OUTPUT_PATH = "models/EDIATH-merged"

# =========================================================
# 🔥 CREATE OUTPUT DIR
# =========================================================
if os.path.exists(OUTPUT_PATH):
    print("⚠️ Removing old merged folder...")
    import shutil
    shutil.rmtree(OUTPUT_PATH)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================================================
# 🔥 FORCE CPU + LOW RAM MODE
# =========================================================
device = "cpu"
dtype = torch.float32

print("🚀 Running LOW-RAM CPU merge mode")
print("========================================")

# =========================================================
# 🔥 VALIDATION
# =========================================================
if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(f"❌ LoRA not found: {LORA_PATH}")

# =========================================================
# 🔥 LOAD BASE MODEL (LOW RAM SAFE)
# =========================================================
print("📥 Loading base model (low RAM mode)...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="cpu",             # ✅ explicit CPU
    low_cpu_mem_usage=True,       # ✅ CRITICAL FIX
    trust_remote_code=True
)

print("✅ Base model loaded")

# =========================================================
# 🔥 LOAD LORA (SAFE)
# =========================================================
print("📥 Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=dtype
)

print("✅ LoRA loaded")

# =========================================================
# 🔥 MERGE
# =========================================================
print("🔀 Merging LoRA...")

try:
    model = model.merge_and_unload()
    print("✅ Merge complete")
except Exception as e:
    print(f"❌ Merge failed: {e}")
    exit(1)

# =========================================================
# 🔥 SAVE MODEL (LOW PRESSURE WRITE)
# =========================================================
print("💾 Saving merged model (low RAM mode)...")

try:
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="500MB"   # ✅ smaller chunks = stable
    )
    print("✅ Model saved")
except Exception as e:
    print(f"❌ Save failed: {e}")
    exit(1)

# =========================================================
# 🔥 SAVE TOKENIZER
# =========================================================
print("💾 Saving tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("✅ Tokenizer saved")
except Exception as e:
    print(f"❌ Tokenizer save failed: {e}")
    exit(1)

# =========================================================
# 🔥 CLEANUP
# =========================================================
del model
torch.cuda.empty_cache()

print("========================================")
print(f"🎉 SUCCESS! Model saved at: {OUTPUT_PATH}")