@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if errorlevel 1 exit /b 1

echo Building vector_add...
nvcc naive_gemm.cu -o naie_gemm.exe
if errorlevel 1 exit /b 1

echo Build complete.