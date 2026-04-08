import os
import subprocess
import sys

# =========================================================
# 🔥 RESOLVE PROJECT ROOT (EDIATH ROOT)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go UP until we find EDIATH root (robust method)
def find_project_root(start_path):
    current = start_path
    for _ in range(5):  # prevent infinite loop
        if os.path.exists(os.path.join(current, "models")):
            return current
        current = os.path.abspath(os.path.join(current, ".."))
    return start_path  # fallback

PROJECT_ROOT = find_project_root(BASE_DIR)

# =========================================================
# 🔥 PATHS (FORCED TO EDIATH/models)
# =========================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(MODELS_DIR, "EDIATH-f16.gguf")
OUTPUT_FILE = os.path.join(MODELS_DIR, "EDIATH-q4_k_m.gguf")

# =========================================================
# 🔥 LLAMA.CPP PATH AUTO-DETECT
# =========================================================
LLAMA_CPP_PATH = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "llama.cpp", "build", "bin", "Release")
)

QUANTIZE_EXE = os.path.join(LLAMA_CPP_PATH, "llama-quantize.exe")

# =========================================================
# 🔥 VALIDATION
# =========================================================
def validate():
    if not os.path.exists(QUANTIZE_EXE):
        raise FileNotFoundError(
            f"\n❌ quantize.exe not found at:\n{QUANTIZE_EXE}\n"
            "👉 Build llama.cpp first:\n"
            "cmake --build . --config Release"
        )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"\n❌ Input GGUF not found:\n{INPUT_FILE}\n"
            "👉 Run convert_to_gguf.py first"
        )

# =========================================================
# 🔥 QUANTIZATION
# =========================================================
def quantize():
    validate()

    print("=================================================")
    print("🚀 EDIATH GGUF QUANTIZATION STARTED")
    print("=================================================")
    print(f"📂 Project Root : {PROJECT_ROOT}")
    print(f"📥 Input File   : {INPUT_FILE}")
    print(f"📤 Output File  : {OUTPUT_FILE}")
    print(f"⚙️ Quantization : Q4_K_M")
    print("=================================================\n")

    command = [
        QUANTIZE_EXE,
        INPUT_FILE,
        OUTPUT_FILE,
        "Q4_K_M"
    ]

    try:
        subprocess.run(command, check=True)

        # ------------------------
        # 🔥 VERIFY OUTPUT
        # ------------------------
        if os.path.exists(OUTPUT_FILE):
            size_gb = os.path.getsize(OUTPUT_FILE) / (1024 ** 3)
            print(f"\n✅ Quantization complete!")
            print(f"📦 Output size: {size_gb:.2f} GB")
            print(f"📁 Saved at: {OUTPUT_FILE}")
        else:
            raise RuntimeError("❌ Output file not created")

    except subprocess.CalledProcessError as e:
        print("\n❌ Quantization failed.")  
        print("👉 Possible reasons:")
        print("- Invalid GGUF input")
        print("- Not enough RAM")
        print("- Broken llama.cpp build")
        print("\nFull error:")
        print(e)
        sys.exit(1)

# =========================================================
# 🔥 ENTRY
# =========================================================
if __name__ == "__main__":
    quantize()