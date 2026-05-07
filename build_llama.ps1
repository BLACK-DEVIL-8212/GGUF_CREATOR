# build_llama.ps1
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Building llama.cpp for GGUF conversion" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Set paths
$LLAMA_PATH = "E:\llama.cpp"
$BUILD_PATH = "$LLAMA_PATH\build"

# Navigate to llama.cpp
Write-Host "`n📂 Navigating to $LLAMA_PATH" -ForegroundColor Yellow
Set-Location $LLAMA_PATH

# Clean build directory if exists
if (Test-Path $BUILD_PATH) {
    Write-Host "🧹 Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Path $BUILD_PATH -Recurse -Force -ErrorAction SilentlyContinue
}

# Create fresh build directory
Write-Host "📁 Creating build directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $BUILD_PATH -Force | Out-Null
Set-Location $BUILD_PATH

# Run CMake configuration
Write-Host "`n🔧 Running CMake configuration..." -ForegroundColor Yellow
Write-Host "   (This may take a minute)" -ForegroundColor Gray

# Try with CUDA first (for RTX 3050)
$cmake_result = cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA=ON 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ CUDA configuration failed, trying without CUDA..." -ForegroundColor Yellow
    $cmake_result = cmake .. -DGGML_CUDA=OFF 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ CMake configuration failed!" -ForegroundColor Red
        Write-Host $cmake_result
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "✅ CMake configuration successful!" -ForegroundColor Green

# Build the project
Write-Host "`n🔨 Building llama.cpp (this will take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host "   Grab a coffee! ☕" -ForegroundColor Gray

$build_result = cmake --build . --config Release -j 4 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host $build_result
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if build succeeded
$QUANTIZE_EXE = "$BUILD_PATH\bin\Release\llama-quantize.exe"
if (Test-Path $QUANTIZE_EXE) {
    Write-Host "`n✅ BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "📁 Quantization tool: $QUANTIZE_EXE" -ForegroundColor White
    Write-Host "================================================" -ForegroundColor Green
    
    # Return to project directory
    Set-Location "E:\GGUF_CREATOR"
    
    Write-Host "`n🚀 Now you can run quantization:" -ForegroundColor Cyan
    Write-Host "   python conversion\quantize.py" -ForegroundColor White
} else {
    Write-Host "`n❌ Build completed but llama-quantize.exe not found!" -ForegroundColor Red
    Write-Host "   Check: $QUANTIZE_EXE" -ForegroundColor Yellow
}

Read-Host "`nPress Enter to exit"