# CUDA/HIP GEMM Optimization Project

A weekend project to build compute-stack depth for NVIDIA/AMD engineering roles:
take matrix multiplication from a naive kernel to a tuned one, benchmark against
the vendor library, then port to HIP/ROCm.

## Step 1: Naive baseline (this scaffold)

```bash
make
./naive_gemm 512      # small N, runs with CPU verification
./naive_gemm 2048      # larger N, timing only (CPU ref would be too slow)
```

This gives you a correctness-checked starting point and a GFLOPS number to beat.
Expect this to be quite slow relative to cuBLAS — that gap is the point.

**If `nvcc` doesn't know your architecture:** check `nvidia-smi` for your GPU model,
look up its SM version (e.g. RTX 30-series = sm_86), and edit `ARCH` in the Makefile.

## Step 2: Shared memory tiling (your turn)

Modify `gemm_naive` (or add a new kernel `gemm_tiled`) so each thread block loads
a `TILE x TILE` sub-block of A and B into `__shared__` memory once, then all
threads in the block reuse it instead of re-reading global memory per element.
Classic tile size to start with: 16x16 or 32x32.

Things to watch for:
- `__syncthreads()` after the load, before the compute
- boundary conditions when N isn't a multiple of TILE
- shared memory bank conflicts (Nsight Compute will flag these)

## Step 3: Profile and tune

```bash
make profile     # requires Nsight Compute (ncu) installed
```

Look at:
- **Achieved occupancy** vs theoretical — low occupancy often means too many
  registers or too much shared memory per block
- **Memory throughput** (should climb a lot after tiling)
- **Warp stall reasons** — tells you what's actually bottlenecking each kernel

Iterate on tile size, add register blocking (each thread computes a 2x2 or 4x4
sub-tile of output instead of 1 element), and re-profile.

## Step 4: Benchmark against cuBLAS

Add a comparison run using `cublasSgemm` on the same input sizes. You won't beat
it (cuBLAS uses tricks like double-buffered shared memory and tensor cores on
newer GPUs) — the goal is quantifying the gap and being able to explain it.

## Step 5: Port to HIP

HIP syntax is nearly identical to CUDA. AMD's `hipify-perl` / `hipify-clang`
tools can auto-convert most of this file:

```bash
hipify-perl naive_gemm.cu > naive_gemm.hip.cpp
```

You'll likely need to fix a few includes and API name differences by hand.
Compare against `rocblas_sgemm` the same way you compared against cuBLAS.

## Step 6: Write up results

Capture a table like:

| Kernel          | N=1024 GFLOPS | vs cuBLAS |
|-----------------|---------------|-----------|
| Naive           |               |           |
| Tiled           |               |           |
| Tiled + tuned   |               |           |
| cuBLAS          |               | 100%      |

Plus 2-3 Nsight Compute screenshots showing the occupancy/memory-throughput
improvement across versions. This table + narrative is what goes on a resume
line or gets pulled up in an interview.
