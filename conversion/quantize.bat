@echo off
set LLAMA_CPP=..\..\llama.cpp
set INPUT=..\models\ediath-f16.gguf
set OUTPUT=..\models\EDIATH-f16.gguf

echo Quantizing model...
%LLAMA_CPP%\quantize.exe %INPUT% %OUTPUT% Q5_K_M
echo Done.