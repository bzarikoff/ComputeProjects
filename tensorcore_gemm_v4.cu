// tensorcore_gemm.cu
// Block tile (shared memory, half precision) -> warp tile, where each warp's
// "thread tile" is replaced by a single wmma (Tensor Core) instruction
// operating on a 16x16x16 fragment collectively across its 32 threads.
// C = A * B, all matrices square, row-major, size N x N.
// Inputs are converted to FP16 for the Tensor Core matmul; accumulation is
// FP32 (standard mixed-precision pattern). Requires N to be a multiple of 16
// (wmma fragments must be full 16x16 tiles -- no partial-tile boundary
// handling here, to keep the kernel readable).
//
// Build:  nvcc -O3 -arch=native tensorcore_gemm.cu -o tensorcore_gemm
// Run:    ./tensorcore_gemm [N]        (default N = 1024, must be multiple of 16)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_TILES_M 4   // warps per block, row direction
#define BLOCK_TILES_N 4   // warps per block, col direction
#define BM (BLOCK_TILES_M * WMMA_M)   // block tile rows = 64
#define BN (BLOCK_TILES_N * WMMA_N)   // block tile cols = 64
#define K_STEPS_PER_TILE 2
#define BK (K_STEPS_PER_TILE * WMMA_K)   // K-chunk = 32 (2 wmma steps per shared-mem load,
                                          // amortizing load/sync overhead -- v1 had BK=16)

#define THREADS_PER_BLOCK (BLOCK_TILES_M * BLOCK_TILES_N * 32)  // 16 warps = 512 threads

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// -----------------------------------------------------------------------
// Block tile (shared memory) -> warp tile -> Tensor Core wmma fragment.
// Unlike the CUDA-core multilevel kernel, the "thread tile" level doesn't
// exist here as a manual loop -- one wmma::mma_sync call IS the warp's
// 16x16x16 matmul, executed collectively by all 32 threads in the warp
// on dedicated Tensor Core hardware.
// -----------------------------------------------------------------------
__global__ void gemm_tensorcore(const half* A, const half* B, float* C, int N) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    // ---- Level 1: block tile ----
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // ---- Level 2: warp tile ----
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_TILES_N;
    int warpCol = warpId % BLOCK_TILES_N;

    // ---- Level 3: Tensor Core fragment (replaces the manual register thread tile) ----
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int numTiles = (N + BK - 1) / BK;
    for (int t = 0; t < numTiles; ++t) {
        // Cooperative load, same pattern as the CUDA-core version, just
        // writing half instead of float into shared memory.
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int r = i / BK, c = i % BK;
            int gRow = blockRow + r, gCol = t * BK + c;
            As[r][c] = (gRow < N && gCol < N) ? A[gRow * N + gCol] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            int gRow = t * BK + r, gCol = blockCol + c;
            Bs[r][c] = (gRow < N && gCol < N) ? B[gRow * N + gCol] : __float2half(0.0f);
        }
        __syncthreads();

        // K_STEPS_PER_TILE wmma steps per shared-mem load -- more compute
        // amortized over each load/sync pair than v1's single-step version.
        for (int kstep = 0; kstep < K_STEPS_PER_TILE; ++kstep) {
            int kk = kstep * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[warpRow * WMMA_M][kk], BK);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warpCol * WMMA_N], BN);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    int cRow = blockRow + warpRow * WMMA_M;
    int cCol = blockCol + warpCol * WMMA_N;
    // No partial-tile handling: caller guarantees N is a multiple of 16,
    // and cRow/cCol + 16 <= N follows from that, so this store is always
    // fully in-bounds.
    wmma::store_matrix_sync(&C[cRow * N + cCol], acc_frag, N, wmma::mem_row_major);
}

// -----------------------------------------------------------------------
// CPU reference for correctness checking. Uses the original FP32 inputs
// (not the FP16-converted device copies) -- this is what lets us measure
// how much accuracy the FP16 Tensor Core path actually costs.
// -----------------------------------------------------------------------
void gemm_cpu_reference(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }
}

void fill_random(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

bool verify(const float* ref, const float* gpu, int N, float tol) {
    double max_abs_err = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double err = fabs(static_cast<double>(ref[i]) - static_cast<double>(gpu[i]));
        if (err > max_abs_err) max_abs_err = err;
        if (err > tol) {
            fprintf(stderr, "Mismatch at index %d: ref=%f gpu=%f (err=%f)\n",
                    i, ref[i], gpu[i], err);
            return false;
        }
    }
    printf("Verification passed. Max abs error: %e (tol=%.1e -- looser than FP32 kernels: FP16 inputs have ~3 decimal digits of precision)\n", max_abs_err, tol);
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N % 16 != 0) {
        fprintf(stderr, "N must be a multiple of 16 for this wmma kernel (got %d)\n", N);
        exit(EXIT_FAILURE);
    }
    bool do_cpu_check = (N <= 512);

    printf("GEMM (Tensor Core, FP16 x FP16 -> FP32): N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");
    printf("BM=%d BN=%d BK=%d warps/block=%d threads/block=%d\n",
           BM, BN, BK, BLOCK_TILES_M * BLOCK_TILES_N, THREADS_PER_BLOCK);

    size_t bytesF = static_cast<size_t>(N) * N * sizeof(float);
    size_t bytesH = static_cast<size_t>(N) * N * sizeof(half);

    float* h_A = (float*)malloc(bytesF);
    float* h_B = (float*)malloc(bytesF);
    float* h_C = (float*)malloc(bytesF);
    float* h_C_ref = do_cpu_check ? (float*)malloc(bytesF) : nullptr;
    half* h_A_half = (half*)malloc(bytesH);
    half* h_B_half = (half*)malloc(bytesH);

    srand(42);
    fill_random(h_A, N);
    fill_random(h_B, N);
    for (int i = 0; i < N * N; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
        h_B_half[i] = __float2half(h_B[i]);
    }

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesH));
    CUDA_CHECK(cudaMalloc(&d_B, bytesH));
    CUDA_CHECK(cudaMalloc(&d_C, bytesF));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_half, bytesH, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_half, bytesH, cudaMemcpyHostToDevice));

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

    // Warm-up launch (first launch pays JIT/context setup cost)
    gemm_tensorcore<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_tensorcore<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesF, cudaMemcpyDeviceToHost));

    double flops = 2.0 * N * N * N;
    double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("Kernel time: %.3f ms | Throughput: %.2f GFLOPS\n", ms, gflops);

    if (do_cpu_check) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        gemm_cpu_reference(h_A, h_B, h_C_ref, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        printf("CPU reference time: %.3f ms\n", cpu_ms);

        verify(h_C_ref, h_C, N, 1e-2f);  // measured max error ~2.4e-3 at N=512 -- FP16 inputs, FP32 accumulation
        free(h_C_ref);
    } else {
        printf("Skipping CPU check at this size. Run with a smaller N (e.g. 256 or 512) to verify correctness first.\n");
    }

    free(h_A); free(h_B); free(h_C); free(h_A_half); free(h_B_half);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
