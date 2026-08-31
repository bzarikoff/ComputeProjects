// naive_gemm.cu
// Step 1 of the GEMM optimization project: naive baseline kernel.
// C = A * B, all matrices square, row-major, size N x N.
//
// Build:  nvcc -O3 -arch=native naive_gemm.cu -o naive_gemm
// Run:    ./naive_gemm [N]        (default N = 1024)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TM 4  // each thread computes TM output rows instead of 1
const static int TILE_SIZE = 8;

#define sharedMemOn 0

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
// Added shared registers and shared memory
// -----------------------------------------------------------------------
__global__ void gemm_thread_tile(const float* A, const float* B, float* C, int N) {
    // Block now covers a (TILE_SIZE*TM) x TILE_SIZE tile of C
    __shared__ float As[TILE_SIZE * TM][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y * TILE_SIZE * TM;
    int blockCol = blockIdx.x * TILE_SIZE;
    int col = blockCol + threadIdx.x;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};  // TM accumulators in registers

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load As: each thread loads TM elements (one per row it owns)
        for (int i = 0; i < TM; ++i) {
            int a_row = blockRow + threadIdx.y + i * TILE_SIZE;
            int a_col = t * TILE_SIZE + threadIdx.x;
            As[threadIdx.y + i * TILE_SIZE][threadIdx.x] =
                (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;
        }

        // Load Bs: unchanged, one element per thread
        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            float b_val = Bs[k][threadIdx.x];   // read from shared mem ONCE
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[threadIdx.y + i * TILE_SIZE][k] * b_val;  // reused TM times
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int row = blockRow + threadIdx.y + i * TILE_SIZE;
        if (row < N && col < N) {
            C[row * N + col] = acc[i];
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

    printf("GEMM: N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");

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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + (TILE_SIZE * TM) - 1) / (TILE_SIZE * TM));

    // Warm-up launch (first launch pays JIT/context setup cost)
    gemm_thread_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_thread_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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
