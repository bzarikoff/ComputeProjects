// cublas_gemm.cu
// Step 4 of the GEMM optimization project: cuBLAS baseline for comparison.
// C = A * B, all matrices square, row-major, size N x N.
//
// Build:  nvcc -O3 -arch=native cublas_gemm.cu -lcublas -o cublas_gemm
// Run:    ./cublas_gemm [N]        (default N = 1024)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                   \
                    status, __FILE__, __LINE__);                            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

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

    printf("cuBLAS GEMM: N = %d, CPU verification: %s\n", N, do_cpu_check ? "on" : "off (N too large)");

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major; our A/B/C are row-major. Row-major C = A*B is
    // numerically identical to column-major C^T = B^T * A^T, and since a
    // row-major NxN buffer IS a column-major NxN buffer of its transpose,
    // we get the right answer by just swapping A and B in the call below --
    // no actual transpose/copy needed.
    // Warm-up launch (first launch pays cuBLAS init / JIT cost)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha,
                              d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha,
                              d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    double flops = 2.0 * N * N * N; // N^3 multiply-adds = 2*N^3 flops
    double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("cuBLAS kernel time: %.3f ms | Throughput: %.2f GFLOPS\n", ms, gflops);

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
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
