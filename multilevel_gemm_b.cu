// multilevel_gemm.cu
// Step: block tile -> warp tile -> thread tile hierarchy, with block shape
// decoupled from tile shape (blockDim is fixed regardless of BM/BN).
// C = A * B, all matrices square, row-major, size N x N.
//
// Build:  nvcc -O3 -arch=native multilevel_gemm.cu -o multilevel_gemm
// Run:    ./multilevel_gemm [N]        (default N = 1024)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define BM 64   // block tile: rows of C this block owns
#define BN 64   // block tile: cols of C this block owns
#define BK 8    // K-dimension chunk size (halved vs baseline -- less shared mem)

#define WM 32   // warp tile: rows of C one warp owns within the block tile
#define WN 32   // warp tile: cols of C one warp owns within the block tile
// BM/WM x BN/WN = 2x2 = 4 warps per block -> blockDim = 4*32 = 128 threads
// (fixed, independent of BM/BN -- decoupled from the tile shape)

#define WARP_ROWS 4   // threads within a warp arranged 4 rows x 8 cols = 32
#define WARP_COLS 8
#define TM 8          // thread tile: each thread computes an 8x4 micro-tile
#define TN 4          // (WARP_ROWS*TM = WM, WARP_COLS*TN = WN)

#define THREADS_PER_BLOCK ((BM / WM) * (BN / WN) * 32)

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
// Multi-level tiled kernel: block tile (shared memory) -> warp tile
// (sub-region within the block) -> thread tile (registers, with a
// register-cache read of As/Bs per k before the TMxTN outer product).
// -----------------------------------------------------------------------
__global__ void gemm_multilevel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // ---- Level 1: block tile ----
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // ---- Level 2: warp tile ----
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);
    int warpTileRow = warpRow * WM;
    int warpTileCol = warpCol * WN;

    // ---- Level 3: thread tile ----
    int threadRow = laneId / WARP_COLS;
    int threadCol = laneId % WARP_COLS;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (N + BK - 1) / BK;
    for (int t = 0; t < numTiles; ++t) {
        // Cooperative load: all threads in the block fill As/Bs together,
        // independent of warp/thread tile assignment.
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int r = i / BK, c = i % BK;
            int gRow = blockRow + r, gCol = t * BK + c;
            As[r][c] = (gRow < N && gCol < N) ? A[gRow * N + gCol] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            int gRow = t * BK + r, gCol = blockCol + c;
            Bs[r][c] = (gRow < N && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a_frag[TM], b_frag[TN];
            for (int i = 0; i < TM; ++i)
                a_frag[i] = As[warpTileRow + threadRow * TM + i][k];
            for (int j = 0; j < TN; ++j)
                b_frag[j] = Bs[k][warpTileCol + threadCol * TN + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = blockRow + warpTileRow + threadRow * TM + i;
            int col = blockCol + warpTileCol + threadCol * TN + j;
            if (row < N && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------
// CPU reference for correctness checking. Slow on purpose -- only used
// to validate, and only worth running at small N (e.g. <= 512) unless
// you want to make coffee while it runs.
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

bool verify(const float* ref, const float* gpu, int N, float tol = 1e-2f) {
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
    printf("Verification passed. Max abs error: %e\n", max_abs_err);
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    bool do_cpu_check = (N <= 512); // CPU reference is O(N^3) and single-threaded

    printf("GEMM (multi-level): N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");
    printf("BM=%d BN=%d BK=%d WM=%d WN=%d TM=%d TN=%d threads/block=%d\n",
           BM, BN, BK, WM, WN, TM, TN, THREADS_PER_BLOCK);

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = do_cpu_check ? (float*)malloc(bytes) : nullptr;

    srand(42);
    fill_random(h_A, N);
    fill_random(h_B, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

    // Warm-up launch (first launch pays JIT/context setup cost)
    gemm_multilevel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_multilevel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    double flops = 2.0 * N * N * N; // N^3 multiply-adds = 2*N^3 flops
    double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("Kernel time: %.3f ms | Throughput: %.2f GFLOPS\n", ms, gflops);

    if (do_cpu_check) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        gemm_cpu_reference(h_A, h_B, h_C_ref, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        printf("CPU reference time: %.3f ms\n", cpu_ms);

        verify(h_C_ref, h_C, N);
        free(h_C_ref);
    } else {
        printf("Skipping CPU check at this size. Run with a smaller N (e.g. 256 or 512) to verify correctness first.\n");
    }

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
