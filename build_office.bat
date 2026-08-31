@echo off

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if errorlevel 1 exit /b 1

echo Building naive...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" naive_gemm.cu -o naive_gemm.exe
if errorlevel 1 exit /b 1

echo Building shared...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" sharedMem_gemm.cu -o sharedMem_gemm.exe
if errorlevel 1 exit /b 1

echo Building shared...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" sharedThreadTile.cu -o sharedThreadTile32.exe
if errorlevel 1 exit /b 1

echo Building shared...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" sharedThreadTile16.cu -o sharedThreadTile16.exe
if errorlevel 1 exit /b 1

echo Building shared...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" sharedThreadTile8.cu -o sharedThreadTile8.exe
if errorlevel 1 exit /b 1

echo Building multilevel...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" multilevel_gemm.cu -o multilevel_gemm.exe
if errorlevel 1 exit /b 1

echo Building tensorcore...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 tensorcore_gemm.cu -o tensorcore_gemm.exe
if errorlevel 1 exit /b 1

echo Building tensorcore v5 (cp.async double-buffered)...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 tensorcore_gemm_v5.cu -o tensorcore_gemm_v5.exe
if errorlevel 1 exit /b 1

echo Building cublas...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" cublas_gemm.cu -lcublas -o cublas_gemm.exe
if errorlevel 1 exit /b 1

echo Build complete.