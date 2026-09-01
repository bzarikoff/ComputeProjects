# GEMM Optimization

A CUDA kernel optimization investigation on an RTX 3080 Ti — from a naive
matrix multiply to ~81–95% of cuBLAS throughput, via shared memory, register
tiling, occupancy analysis, warp-level tiling, Tensor Cores, and asynchronous
double buffering.

**[Open the full interactive report](./gemm_report.html)** — includes the
occupancy charts, formulas, and hover tooltips referenced below.

## Results

| Kernel | GFLOPS | vs cuBLAS |
|---|---:|---:|
| Naive | 2,070 | 14.7% |
| Shared memory tiled | 2,699 | 19.2% |
| Register tiled — `TILE_SIZE=8` | 4,025 | 28.6% |
| Register tiled — `TILE_SIZE=32` | 5,021 | 35.7% |
| Register tiled — `TILE_SIZE=16` (best CUDA-core) | 5,800 | 41.0% |
| Multi-level tiled (block/warp/thread) | 5,592 | 39.7% |
| Tensor Core, single-buffered | 8,224 | 58.4% |
| **Tensor Core, `cp.async` double-buffered** | **13,358** | **~81–95%** |
| cuBLAS (`cublasSgemm`) | 14,081–16,513 | 100% |

All figures at N=1024. Correctness verified against a CPU reference at
N≤512. cuBLAS varied across runs (see *Why cuBLAS was variable* below), so
the double-buffered kernel's ratio is quoted as a range rather than a single
number.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti (GA102, compute capability 8.6) |
| SMs | 80 |
| Max threads / SM | 1536 (48 warps) |
| Registers / SM | 65,536 |
| Shared memory / block | 49,152 bytes (default), 100 KB / SM |
| Peak memory bandwidth | 912.1 GB/s (384-bit @ 9501 MHz) |
| Driver / CUDA | 572.83 / 12.8 |

## Methodology

1. **Naive kernel** — one thread per output element, no data reuse.
   Establishes the correctness baseline and the number to beat.
2. **Shared-memory tiling** — each block cooperatively loads a tile of A/B
   into `__shared__` memory once, reused by every thread in the block.
3. **Register (thread) tiling + tile-size sweep** — each thread accumulates
   `TM=4` output rows in registers. Swept `TILE_SIZE` ∈ {8, 16, 32} to find
   the occupancy sweet spot (see *Occupancy investigation* below).
4. **Multi-level (block → warp → thread) tiling** — added an explicit
   warp-tile layer, decoupling block shape from tile shape. On CUDA cores
   alone this reorganized indices without adding new reuse (see *Findings*).
5. **Tensor Cores (FP16×FP16→FP32)** — replaced the manual thread-tile
   accumulation with `wmma::mma_sync`, a warp-collective 16×16×16 matrix
   instruction on dedicated hardware.
6. **`cp.async` double buffering** — two shared-memory buffers instead of
   one: the next tile loads asynchronously while Tensor Cores compute on the
   current tile, instead of load → sync → compute → sync running serially.

## Occupancy investigation

Before adding warp-level tiling, the register-tiled kernel (`TM=4`) was
swept across `TILE_SIZE` ∈ {8, 16, 32} — the value that sets both the
shared-memory tile size *and* the block's thread count.

`TILE_SIZE=16` won at 5,800 GFLOPS, beating both neighbors (4,025 and 5,021
GFLOPS). Why:

- **`TILE_SIZE=32`** is bound by raw thread count: `blockDim(32,32) = 1024`
  threads/block, and since `2×1024 > 1536` max threads/SM, only one block
  can ever be resident — a hard ceiling regardless of registers or shared
  memory.
- **`TILE_SIZE=8`** is bound by a flatter constraint: the GPU's hardware
  cap of 16 resident blocks/SM, which binds specifically because 64-thread
  blocks are small enough that every other resource would allow far more
  than 16.
- **`TILE_SIZE=16`** lands in the gap between those two failure modes —
  small enough that the flat 16-block cap isn't binding, large enough that
  thread count isn't binding either — leaving registers as the binding
  constraint, at a noticeably better ratio than either neighbor.

The full interactive charts (resident warps by limiting resource, and
resulting occupancy vs. tile size, both with hover tooltips) plus the exact
formulas used are in **[gemm_report.html](./gemm_report.html)**.

## Key findings

- **Bigger tiles are not automatically better.** `TILE_SIZE=16` beat
  `TILE_SIZE=32` by 15% despite computing a smaller shared-memory tile per
  block — the bottleneck moved from memory traffic to occupancy, and
  occupancy is governed by whichever hardware resource binds first, not by
  tile size alone.
- **Warp tiling alone did nothing on CUDA cores.** Adding an explicit
  warp-tile layer reorganizes *which* thread reads *which* shared-memory
  address — it creates no new reuse by itself. The best multi-level
  configuration (5,592 GFLOPS) landed within noise of the simpler two-level
  kernel (5,800 GFLOPS). Warp tiling only pays off when paired with
  something that actually needs warp granularity: bank-conflict-aware
  layouts, `__shfl_sync`, or — as it turned out — Tensor Cores.
- **Tensor Cores were the single largest architectural jump.** Replacing
  the manual register-accumulation loop with one `wmma::mma_sync` call per
  warp lifted throughput from 5,800 to 8,224 GFLOPS.
- **Double buffering beat higher occupancy.** `cp.async` double buffering
  dropped occupancy from 100% to ~67% (two shared-memory buffers instead of
  one costs registers) yet delivered the largest single gain of the
  investigation — 8,224 → 13,358 GFLOPS. Overlapping the next tile's load
  with the current tile's compute eliminated far more idle time than the
  occupancy loss cost.
- **Empirical tuning beat theoretical prediction, repeatedly.** Every stage
  that looked like a clear win on paper (bigger tiles, more register reuse,
  warp tiling) had to be measured to find out whether it actually helped —
  several didn't. This mirrors how production libraries like cuBLAS and
  rocBLAS work: an offline empirical search over kernel variants per
  problem shape, not a single hand-derived "optimal" kernel.

## Why cuBLAS was variable

cuBLAS's measured throughput ranged 14,081–16,513 GFLOPS across runs, more
than our own kernels varied. Two compounding reasons:

1. cuBLAS ran in ~130–150 microseconds at N=1024 — dramatically shorter
   than our own kernels (0.3–0.6ms). Timing jitter from GPU clock
   granularity and driver/launch overhead is a much larger relative
   fraction of a very short measurement.
2. This is a shared consumer GPU, not a dedicated benchmarking machine —
   background desktop compositing, browser, and notification processes were
   concurrently active during benchmarking, and any brief contention has an
   outsized effect on a sub-millisecond kernel launch.

## Remaining gap to cuBLAS

The final kernel reaches roughly 81–95% of cuBLAS. The likely remainder:
cuBLAS dispatches between many pre-tuned kernel variants selected per
problem shape rather than running one fixed kernel; it likely uses deeper
pipelines (3+ stages vs. this investigation's 2); and portions of its
kernels may be hand-scheduled at the assembly (SASS) level in ways standard
CUDA C++ can't directly express. Closing that gap further means adopting
those same techniques more exhaustively, not a new category of
optimization.

## Files

| File | Description |
|---|---|
| `naive_gemm.cu` | Step 1 baseline |
| `sharedMem_gemm.cu` | Step 2, shared-memory tiling |
| `sharedThreadTile8.cu` / `16.cu` / `32.cu` | Step 3, register tiling at each swept `TILE_SIZE` |
| `multilevel_gemm.cu` | Block/warp/thread tiling sweep |
| `tensorcore_gemm.cu` (v1) → `tensorcore_gemm_v5.cu` | Tensor Core progression, ending in `cp.async` double buffering |
| `cublas_gemm.cu` | cuBLAS comparison harness |
| `build_office.bat` | Builds every kernel above |
| `info.txt` | Working notes: GPU specs, occupancy math, register-count verification recipe |
| `gemm_report.html` | The full illustrated report (this file, with interactive charts) |
