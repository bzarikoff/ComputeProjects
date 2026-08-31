// tensorcore_gemm_v5.cu
// Same block/warp/Tensor-Core structure as tensorcore_gemm.cu (v1), but the
// global->shared loads use cp.async double buffering: while the warps compute
// on buffer A, the next tile is already being copied into buffer B in the
// background, instead of load -> sync -> compute -> sync happening serially.
// C = A * B, all matrices square, row-major, size N x N.
// Requires N to be a multiple of BM (=64) -- no partial-tile handling, and
// no per-element boundary zero-fill (cp.async copies fixed-size chunks).
//
// Build:  nvcc -O3 -arch=sm_86 tensorcore_gemm_v5.cu -o tensorcore_gemm_v5
// Run:    ./tensorcore_gemm_v5 [N]        (default N = 1024, must be multiple of 64)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_TILES_M 4
#define BLOCK_TILES_N 4
#define BM (BLOCK_TILES_M * WMMA_M)   // 64
#define BN (BLOCK_TILES_N * WMMA_N)   // 64
#define BK WMMA_K                      // 16

#define THREADS_PER_BLOCK (BLOCK_TILES_M * BLOCK_TILES_N * 32)  // 512

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
// Double-buffered cp.async load of one (A tile, B tile) pair into buffer
// `buf`. Copies 2 halves (4 bytes) at a time per thread -- BM*(BK/2) and
// BK*(BN/2) both equal THREADS_PER_BLOCK here, so each thread issues exactly
// one 4-byte async copy per array, no stride loop needed.
// -----------------------------------------------------------------------
__device__ __forceinline__ void issue_tile_load(half (*As)[BM][BK], half (*Bs)[BK][BN],
                                                 int buf, const half* A, const half* B,
                                                 int N, int blockRow, int blockCol, int tileK) {
    int pairIdxA = threadIdx.x;
    int rA = pairIdxA / (BK / 2);
    int cA = (pairIdxA % (BK / 2)) * 2;
    int gRowA = blockRow + rA, gColA = tileK + cA;
    __pipeline_memcpy_async(&As[buf][rA][cA], &A[gRowA * N + gColA], sizeof(half) * 2);

    int pairIdxB = threadIdx.x;
    int rB = pairIdxB / (BN / 2);
    int cB = (pairIdxB % (BN / 2)) * 2;
    int gRowB = tileK + rB, gColB = blockCol + cB;
    __pipeline_memcpy_async(&Bs[buf][rB][cB], &B[gRowB * N + gColB], sizeof(half) * 2);

    __pipeline_commit();
}

__global__ void gemm_tensorcore_v5(const half* A, const half* B, float* C, int N) {
    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_TILES_N;
    int warpCol = warpId % BLOCK_TILES_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int numTiles = N / BK;

    // Prologue: kick off the first tile's load before the loop starts.
    int buf = 0;
    issue_tile_load(As, Bs, buf, A, B, N, blockRow, blockCol, 0);

    for (int t = 0; t < numTiles; ++t) {
        int nextBuf = 1 - buf;
        if (t + 1 < numTiles) {
            // Start the NEXT tile's load now, while this iteration's compute
            // (below) runs concurrently on the ALREADY-loaded current buffer.
            issue_tile_load(As, Bs, nextBuf, A, B, N, blockRow, blockCol, (t + 1) * BK);
        }

        // Wait for THIS thread's own current-tile copies (issued last
        // iteration, or in the prologue) to land, then barrier so every
        // thread in the block agrees the whole tile is visible.
        __pipeline_wait_prior(t + 1 < numTiles ? 1 : 0);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &As[buf][warpRow * WMMA_M][0], BK);
        wmma::load_matrix_sync(b_frag, &Bs[buf][0][warpCol * WMMA_N], BN);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // Barrier before the buffer we just read gets overwritten by next
        // iteration's async copy (nextBuf becomes buf one iteration later).
        __syncthreads();
        buf = nextBuf;
    }

    int cRow = blockRow + warpRow * WMMA_M;
    int cCol = blockCol + warpCol * WMMA_N;
    wmma::store_matrix_sync(&C[cRow * N + cCol], acc_frag, N, wmma::mem_row_major);
}

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
    printf("Verification passed. Max abs error: %e (tol=%.1e)\n", max_abs_err, tol);
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N % BM != 0 || N % BN != 0) {
        fprintf(stderr, "N must be a multiple of %d for this kernel (got %d)\n", BM, N);
        exit(EXIT_FAILURE);
    }
    bool do_cpu_check = (N <= 512);

    printf("GEMM (Tensor Core v5, cp.async double-buffered): N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");
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
    dim3 gridDim(N / BN, N / BM);

    gemm_tensorcore_v5<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_tensorcore_v5<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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

        verify(h_C_ref, h_C, N, 1e-2f);
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
