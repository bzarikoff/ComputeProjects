@echo off

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if errorlevel 1 exit /b 1

echo Building vector_add...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" naive_gemm.cu -o naive_gemm.exe
if errorlevel 1 exit /b 1

echo Build complete.