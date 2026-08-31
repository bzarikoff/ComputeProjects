// tensorcore_gemm_v2.cu
// Same block tile -> warp tile -> Tensor Core structure as tensorcore_gemm.cu,
// but each warp now owns a WARP_TILES_M x WARP_TILES_N grid of wmma fragments
// (fragment-level reuse, the tensor-core analogue of TM/TN register tiling),
// and BK spans multiple WMMA_K steps per shared-memory load/sync cycle.
// C = A * B, all matrices square, row-major, size N x N.
// Requires N to be a multiple of BM/BN (both 128 here) for this simple
// version -- no partial-tile boundary handling.
//
// Build:  nvcc -O3 -arch=native tensorcore_gemm_v2.cu -o tensorcore_gemm_v2
// Run:    ./tensorcore_gemm_v2 [N]        (default N = 1024, must be multiple of 128)

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

#define WARP_TILES_M 2   // fragments per warp, row direction
#define WARP_TILES_N 2   // fragments per warp, col direction
#define WM (WARP_TILES_M * WMMA_M)   // warp tile rows = 32
#define WN (WARP_TILES_N * WMMA_N)   // warp tile cols = 32

#define BLOCK_TILES_M 2   // warps per block, row direction (reduced from v2's 4:
#define BLOCK_TILES_N 2   // v2's 128x128 block tile shrank the grid below the
                           // GPU's 80 SMs at N=1024/512 -- keep block tile at
                           // 64x64 like v1, still with 2x2 fragment reuse/warp)
#define BM (BLOCK_TILES_M * WM)   // block tile rows = 64
#define BN (BLOCK_TILES_N * WN)   // block tile cols = 64

#define K_STEPS_PER_TILE 2
#define BK (K_STEPS_PER_TILE * WMMA_K)   // K-chunk = 32 (2 wmma steps per shared-mem load)

#define THREADS_PER_BLOCK (BLOCK_TILES_M * BLOCK_TILES_N * 32)   // 4 warps = 128 threads

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void gemm_tensorcore_v3(const half* A, const half* B, float* C, int N) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / BLOCK_TILES_N;
    int warpCol = warpId % BLOCK_TILES_N;
    int warpTileRow = warpRow * WM;
    int warpTileCol = warpCol * WN;

    // WARP_TILES_M x WARP_TILES_N accumulator fragments per warp --
    // fragment-level reuse, the tensor-core analogue of TM/TN register tiling.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_TILES_M][WARP_TILES_N];
    for (int i = 0; i < WARP_TILES_M; ++i)
        for (int j = 0; j < WARP_TILES_N; ++j)
            wmma::fill_fragment(acc_frag[i][j], 0.0f);

    int numTiles = (N + BK - 1) / BK;
    for (int t = 0; t < numTiles; ++t) {
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

        for (int kstep = 0; kstep < K_STEPS_PER_TILE; ++kstep) {
            int kk = kstep * WMMA_K;

            // Load each fragment ONCE per k-step, reuse across the WARP_TILES_M x
            // WARP_TILES_N outer product below -- same register-cache idea as
            // a_frag/b_frag in the CUDA-core multilevel kernel.
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILES_N];

            for (int i = 0; i < WARP_TILES_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[warpTileRow + i * WMMA_M][kk], BK);
            for (int j = 0; j < WARP_TILES_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][warpTileCol + j * WMMA_N], BN);

            for (int i = 0; i < WARP_TILES_M; ++i)
                for (int j = 0; j < WARP_TILES_N; ++j)
                    wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < WARP_TILES_M; ++i) {
        for (int j = 0; j < WARP_TILES_N; ++j) {
            int cRow = blockRow + warpTileRow + i * WMMA_M;
            int cCol = blockCol + warpTileCol + j * WMMA_N;
            wmma::store_matrix_sync(&C[cRow * N + cCol], acc_frag[i][j], N, wmma::mem_row_major);
        }
    }
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

    printf("GEMM (Tensor Core v3, 64x64 block + fragment tiling): N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");
    printf("BM=%d BN=%d BK=%d WM=%d WN=%d WARP_TILES=%dx%d warps/block=%d threads/block=%d\n",
           BM, BN, BK, WM, WN, WARP_TILES_M, WARP_TILES_N, BLOCK_TILES_M * BLOCK_TILES_N, THREADS_PER_BLOCK);

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

    gemm_tensorcore_v3<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_tensorcore_v3<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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
