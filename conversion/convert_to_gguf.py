import os
import subprocess
import sys

# ==========================================================
# EDIATH GGUF CONVERTER
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MERGED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "EDIATH-merged")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "models", "EDIATH-f16.gguf")

# Expect llama.cpp to be next to EDIATH_CORE
LLAMA_CPP_PATH = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "llama.cpp")
)

def check_paths():
    if not os.path.isdir(MERGED_MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ Merged model folder not found:\n{MERGED_MODEL_PATH}\n"
            "Run merge_lora.py first."
        )

    if not os.path.isdir(LLAMA_CPP_PATH):
        raise FileNotFoundError(
            f"\n❌ llama.cpp not found at:\n{LLAMA_CPP_PATH}\n\n"
            "Expected structure:\n"
            "Desktop/\n"
            " ├── llama.cpp\n"
            " └── EDIATH_CORE\n\n"
            "Clone it with:\n"
            "git clone https://github.com/ggerganov/llama.cpp"
        )

    convert_script = os.path.join(
        LLAMA_CPP_PATH,
        "convert_hf_to_gguf.py"
    )

    if not os.path.exists(convert_script):
        raise FileNotFoundError(
            f"\n❌ convert_hf_to_gguf.py not found:\n{convert_script}\n\n"
            "Update llama.cpp:\n"
            "cd llama.cpp\n"
            "git pull\n"
            "pip install -r requirements.txt"
        )

    return convert_script


def convert():
    print("🔍 Checking paths...")
    convert_script = check_paths()

    print("\n🚀 Starting GGUF conversion")
    print(f"📂 Input  : {MERGED_MODEL_PATH}")
    print(f"💾 Output : {OUTPUT_FILE}")
    print(f"🛠 llama.cpp : {LLAMA_CPP_PATH}")
    print("\nUsing quantization: q4_k_m (Recommended for RTX 3050)\n")

    command = [
        sys.executable,
        convert_script,
        MERGED_MODEL_PATH,
        "--outfile",
        OUTPUT_FILE,
        "--outtype",
        "f16"
    ]

    try:
        subprocess.run(command, check=True)
        print("\n✅ GGUF conversion completed successfully.")
        print(f"📦 File saved at:\n{OUTPUT_FILE}")

    except subprocess.CalledProcessError as e:
        print("\n❌ Conversion failed.")
        print("Common reasons:")
        print(" - llama.cpp is outdated (run git pull)")
        print(" - DeepSeek architecture not supported")
        print(" - Not enough RAM")
        print("\nFull error:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    convert()