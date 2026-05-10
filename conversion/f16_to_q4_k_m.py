# conversion/f16_to_q4_k_m.py

import subprocess
from pathlib import Path
import sys


# =========================================================
# AUTO PATH DETECTION
# =========================================================

# AI_MODAL_GGUF_CREATOR/
ROOT_DIR = Path(__file__).resolve().parent.parent

# E:\
BASE_DIR = ROOT_DIR.parent

# llama.cpp/
LLAMA_CPP_PATH = BASE_DIR / "llama.cpp"

# models/
MODELS_DIR = ROOT_DIR / "models"

# =========================================================
# MODEL FILES
# =========================================================

INPUT_MODEL = MODELS_DIR / "EDIATH-f16.gguf"
OUTPUT_MODEL = MODELS_DIR / "EDIATH-q4_k_m.gguf"


# =========================================================
# FIND QUANTIZE EXECUTABLE
# =========================================================

def find_quantize_executable():

    possible_paths = [

        # Latest llama.cpp builds
        LLAMA_CPP_PATH / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_PATH / "build" / "bin" / "Release" / "quantize.exe",

        # Older builds
        LLAMA_CPP_PATH / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_PATH / "build" / "bin" / "quantize.exe",

        # Root executables
        LLAMA_CPP_PATH / "llama-quantize.exe",
        LLAMA_CPP_PATH / "quantize.exe",
    ]

    for exe in possible_paths:
        if exe.exists():
            return exe

    return None


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("EDIATH F16 → Q4_K_M GGUF QUANTIZER")
    print("=" * 65)

    print(f"\n[ROOT]")
    print(ROOT_DIR)

    print(f"\n[LLAMA.CPP]")
    print(LLAMA_CPP_PATH)

    print(f"\n[INPUT MODEL]")
    print(INPUT_MODEL)

    print(f"\n[OUTPUT MODEL]")
    print(OUTPUT_MODEL)

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not INPUT_MODEL.exists():

        print("\n[ERROR] F16 model not found.")
        print("\nExpected file:")
        print(INPUT_MODEL)

        sys.exit(1)

    # =====================================================
    # FIND QUANTIZER
    # =====================================================

    quantize_exe = find_quantize_executable()

    if quantize_exe is None:

        print("\n[ERROR] llama.cpp quantizer not found.")

        print("\nExpected one of these:")

        print(LLAMA_CPP_PATH / "build/bin/Release/llama-quantize.exe")
        print(LLAMA_CPP_PATH / "build/bin/Release/quantize.exe")
        print(LLAMA_CPP_PATH / "llama-quantize.exe")

        print("\nBuild llama.cpp first.")

        sys.exit(1)

    print(f"\n[QUANTIZER FOUND]")
    print(quantize_exe)

    # =====================================================
    # COMMAND
    # =====================================================

    command = [
        str(quantize_exe),
        str(INPUT_MODEL),
        str(OUTPUT_MODEL),
        "Q4_K_M"
    ]

    print("\n[RUNNING COMMAND]\n")
    print(" ".join(command))

    print("\n[STARTING QUANTIZATION]\n")

    # =====================================================
    # EXECUTE
    # =====================================================

    try:

        subprocess.run(
            command,
            check=True
        )

        print("\n" + "=" * 65)
        print("[SUCCESS] QUANTIZATION COMPLETED")
        print("=" * 65)

        if OUTPUT_MODEL.exists():

            size_gb = OUTPUT_MODEL.stat().st_size / (1024 ** 3)

            print(f"\n[MODEL SIZE]")
            print(f"{size_gb:.2f} GB")

            print(f"\n[SAVED TO]")
            print(OUTPUT_MODEL)

        else:

            print("\n[WARNING] Output model not found.")

    except subprocess.CalledProcessError as e:

        print("\n[ERROR] Quantization failed.")
        print(e)

        sys.exit(1)

    except Exception as e:

        print("\n[ERROR] Unexpected error.")
        print(str(e))

        sys.exit(1)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()