#!/usr/bin/env python3
"""
GGUF Quantization Script for EDIATH Model
Converts merged Mistral-7B model to GGUF format for CPU inference
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================
# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "EDIATH-merged")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
LLAMA_CPP_PATH = "E:/llama.cpp"

# Quantization types
QUANTIZATION_TYPES = {
    "Q2_K": "Smallest size, lowest quality (2-3 GB)",
    "Q3_K": "Very small size, low quality (3-4 GB)",
    "Q4_K_M": "Small size, good quality (4-5 GB) - RECOMMENDED",
    "Q5_K_M": "Medium size, high quality (5-6 GB)",
    "Q6_K": "Large size, very high quality (6-7 GB)",
    "Q8_0": "Very large size, near original quality (7-8 GB)",
}

DEFAULT_QUANT = "Q4_K_M"

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def print_colored(message, color="white"):
    """Print colored text (Windows compatible)"""
    colors = {
        "red": 31, "green": 32, "yellow": 33, "blue": 34, 
        "magenta": 35, "cyan": 36, "white": 37
    }
    if platform.system() == "Windows" and hasattr(sys.stdout, "isatty"):
        print(f"\033[{colors.get(color, 37)}m{message}\033[0m")
    else:
        print(message)

def print_progress(message, status="info"):
    """Pretty print progress messages"""
    symbols = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️", "progress": "🔄"}
    print(f"{symbols.get(status, '📌')} {message}")

def find_llama_quantize():
    """Find llama-quantize executable in common locations"""
    possible_paths = [
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "Release", "llama-quantize.exe"),
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize.exe"),
        os.path.join(LLAMA_CPP_PATH, "llama-quantize.exe"),
        "llama-quantize.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def find_convert_script():
    """Find convert script in llama.cpp"""
    possible_paths = [
        os.path.join(LLAMA_CPP_PATH, "convert-hf-to-gguf.py"),
        os.path.join(LLAMA_CPP_PATH, "convert.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def check_model():
    """Check if merged model exists"""
    if not os.path.exists(MODEL_PATH):
        print_progress(f"Model not found: {MODEL_PATH}", "error")
        print_progress("Please run merge_lora.py first", "warning")
        return False
    
    required = ["config.json", "tokenizer.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_PATH, f))]
    
    if missing:
        print_progress(f"Missing model files: {missing}", "error")
        return False
    
    print_progress(f"Model found: {MODEL_PATH}", "success")
    return True

# =====================================================
# BUILD LLAMA.CPP
# =====================================================

def build_llama_cpp():
    """Build llama.cpp from source"""
    print_progress("Building llama.cpp...", "progress")
    
    # Check if CMake is installed
    try:
        subprocess.run(["cmake", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_progress("CMake not found! Please install CMake first:", "error")
        print("   Download from: https://cmake.org/download/")
        print("   Or run: winget install Kitware.CMake")
        return False
    
    # Clone llama.cpp if not exists
    if not os.path.exists(LLAMA_CPP_PATH):
        print_progress(f"Cloning llama.cpp to {LLAMA_CPP_PATH}...", "progress")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_PATH],
                check=True
            )
        except Exception as e:
            print_progress(f"Failed to clone: {e}", "error")
            return False
    
    # Create build directory
    build_dir = os.path.join(LLAMA_CPP_PATH, "build")
    if os.path.exists(build_dir):
        import shutil
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    
    # Run CMake
    print_progress("Running CMake configuration...", "progress")
    try:
        subprocess.run(
            ["cmake", "..", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DGGML_CUDA=ON"],
            cwd=build_dir,
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError:
        print_progress("CUDA version failed, trying without CUDA...", "warning")
        subprocess.run(
            ["cmake", "..", "-DGGML_CUDA=OFF"],
            cwd=build_dir,
            check=True
        )
    
    # Build
    print_progress("Building (this will take 5-10 minutes)...", "progress")
    try:
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j", "4"],
            cwd=build_dir,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print_progress(f"Build failed: {e}", "error")
        return False
    
    # Verify build
    quantize_exe = find_llama_quantize()
    if quantize_exe:
        print_progress(f"Build successful! Found: {quantize_exe}", "success")
        return True
    else:
        print_progress("Build completed but llama-quantize.exe not found", "error")
        return False

# =====================================================
# CONVERSION FUNCTIONS
# =====================================================

def convert_to_gguf():
    """Convert HuggingFace model to GGUF"""
    print_progress("Converting to GGUF format...", "progress")
    
    convert_script = find_convert_script()
    if not convert_script:
        print_progress("Convert script not found", "error")
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "ediath-model-f16.gguf")
    
    cmd = [
        sys.executable, convert_script,
        "--model", MODEL_PATH,
        "--outfile", output_file,
        "--outtype", "f16"
    ]
    
    try:
        print_progress(f"Running conversion...", "info")
        subprocess.run(cmd, check=True)
        print_progress(f"Conversion complete: {output_file}", "success")
        return output_file
    except Exception as e:
        print_progress(f"Conversion failed: {e}", "error")
        return None

def quantize_gguf(input_file, quant_type):
    """Quantize GGUF file"""
    print_progress(f"Quantizing to {quant_type}...", "progress")
    
    quantize_exe = find_llama_quantize()
    if not quantize_exe:
        return None
    
    output_file = os.path.join(OUTPUT_DIR, f"ediath-model-{quant_type.lower()}.gguf")
    
    cmd = [quantize_exe, input_file, output_file, quant_type]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Show compression stats
        in_size = os.path.getsize(input_file) / (1024**3)
        out_size = os.path.getsize(output_file) / (1024**3)
        
        print_progress(f"Quantization complete!", "success")
        print(f"   📊 Input:  {in_size:.2f} GB")
        print(f"   📊 Output: {out_size:.2f} GB")
        print(f"   📊 Saved:  {in_size - out_size:.2f} GB ({(1 - out_size/in_size)*100:.1f}%)")
        
        return output_file
    except Exception as e:
        print_progress(f"Quantization failed: {e}", "error")
        return None

def test_model(model_path):
    """Test the quantized model"""
    print_progress("Testing model...", "progress")
    
    # Find llama-cli
    cli_paths = [
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "Release", "llama-cli.exe"),
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-cli.exe"),
        "llama-cli.exe",
    ]
    
    llama_cli = None
    for path in cli_paths:
        if os.path.exists(path):
            llama_cli = path
            break
    
    if not llama_cli:
        print_progress("llama-cli not found, skipping test", "warning")
        return
    
    prompt = "What is artificial intelligence? Explain briefly."
    
    cmd = [
        llama_cli,
        "-m", model_path,
        "-p", prompt,
        "-n", "100",
        "-ngl", "24"
    ]
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing with prompt: {prompt}")
        print(f"{'='*60}\n")
        subprocess.run(cmd, check=True)
        print(f"\n{'='*60}")
        print_progress("Test completed!", "success")
    except Exception as e:
        print_progress(f"Test failed: {e}", "warning")

# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    print("\n" + "="*60)
    print_colored("🚀 GGUF Model Quantization Tool", "cyan")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Step 1: Check model
    if not check_model():
        return False
    
    # Step 2: Check/install llama.cpp
    quantize_exe = find_llama_quantize()
    if not quantize_exe:
        print_progress("llama-quantize.exe not found", "warning")
        print("\nDo you want to build llama.cpp automatically?")
        print("This will:")
        print("  - Clone llama.cpp to E:\\llama.cpp")
        print("  - Build it (requires CMake and Visual Studio)")
        print("  - Take 5-10 minutes")
        
        build_choice = input("\nBuild llama.cpp? (y/n): ").strip().lower()
        if build_choice == 'y':
            if not build_llama_cpp():
                return False
        else:
            print_progress("Cannot continue without llama.cpp", "error")
            print("\nPlease build manually:")
            print("  1. cd E:\\")
            print("  2. git clone https://github.com/ggerganov/llama.cpp.git")
            print("  3. cd llama.cpp")
            print("  4. mkdir build && cd build")
            print("  5. cmake .. -DGGML_CUDA=OFF")
            print("  6. cmake --build . --config Release -j 4")
            return False
    
    # Step 3: Show quantization options
    print("\n📊 Quantization Options:")
    print("-" * 55)
    for quant, desc in QUANTIZATION_TYPES.items():
        marker = " ← RECOMMENDED" if quant == DEFAULT_QUANT else ""
        print(f"  {quant}: {desc}{marker}")
    print("-" * 55)
    
    # Step 4: Get user choice
    quant_type = input(f"\nEnter quantization type [default: {DEFAULT_QUANT}]: ").strip().upper()
    if not quant_type:
        quant_type = DEFAULT_QUANT
    
    if quant_type not in QUANTIZATION_TYPES:
        print_progress(f"Invalid choice, using {DEFAULT_QUANT}", "warning")
        quant_type = DEFAULT_QUANT
    
    print_progress(f"Using: {quant_type} - {QUANTIZATION_TYPES[quant_type]}", "info")
    
    # Step 5: Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 6: Convert
    fp16_file = convert_to_gguf()
    if not fp16_file:
        return False
    
    # Step 7: Quantize
    quantized_file = quantize_gguf(fp16_file, quant_type)
    if not quantized_file:
        return False
    
    # Step 8: Test (optional)
    print()
    test_choice = input("Test the quantized model? (y/n): ").strip().lower()
    if test_choice == 'y':
        test_model(quantized_file)
    
    # Final success message
    print("\n" + "="*60)
    print_colored("🎉 QUANTIZATION COMPLETE!", "green")
    print("="*60)
    print(f"📁 Model: {quantized_file}")
    print(f"📊 Size: {os.path.getsize(quantized_file) / (1024**3):.2f} GB")
    print(f"📊 Type: {quant_type}")
    print("\n🔧 Quick test command:")
    print(f"   cd {LLAMA_CPP_PATH}\\build\\bin\\Release")
    print(f"   ./llama-cli.exe -m {quantized_file} -p 'Hello' -n 50")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
