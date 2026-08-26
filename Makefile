# Adjust -arch to match your GPU if `native` isn't supported by your nvcc version:
#   Ampere (30xx/A100)   -> sm_86 / sm_80
#   Ada (40xx)           -> sm_89
#   Hopper (H100)        -> sm_90
NVCC := nvcc
ARCH := -arch=native
FLAGS := -O3 $(ARCH)

all: naive_gemm

naive_gemm: naive_gemm.cu
	$(NVCC) $(FLAGS) naive_gemm.cu -o naive_gemm

profile: naive_gemm
	ncu --set full -o naive_gemm_profile ./naive_gemm 1024

clean:
	rm -f naive_gemm *.ncu-rep naive_gemm_profile.*

.PHONY: all profile clean
