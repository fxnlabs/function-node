#include "matmul.h"
#include "cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// CPU matrix multiplication for verification
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify results
bool verify_results(const float* C_gpu, const float* C_cpu, int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C_gpu[i] - C_cpu[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f, diff = %f\n",
                   i, C_gpu[i], C_cpu[i], diff);
            return false;
        }
    }
    return true;
}

// Test matrix multiplication with given dimensions
void test_matmul(int M, int N, int K) {
    printf("\nTesting matrix multiplication: %d x %d x %d\n", M, N, K);
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(42);  // Fixed seed for reproducibility
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Create timer
    CudaTimer timer;
    createTimer(&timer);
    
    // Warm-up run
    matmul_cuda(h_A, h_B, h_C_gpu, M, N, K);
    
    // Timed run
    startTimer(&timer);
    cudaError_t error = matmul_cuda(h_A, h_B, h_C_gpu, M, N, K);
    stopTimer(&timer);
    
    if (error != cudaSuccess) {
        printf("CUDA matrix multiplication failed: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    printf("GPU Time: %.3f ms\n", timer.elapsed_ms);
    
    // Calculate performance metrics
    {
        float gflops = calculateGFLOPS(M, N, K, timer.elapsed_ms);
        size_t total_bytes = size_A + size_B + size_C;
        float bandwidth = calculateBandwidth(total_bytes, timer.elapsed_ms);
        
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective Bandwidth: %.2f GB/s\n", bandwidth);
    }
    
    // Verify results for small matrices
    if (M <= 512 && N <= 512 && K <= 512) {
        printf("Verifying results...\n");
        clock_t cpu_start = clock();
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        clock_t cpu_end = clock();
        float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;
        printf("CPU Time: %.3f ms\n", cpu_time);
        printf("Speedup: %.2fx\n", cpu_time / timer.elapsed_ms);
        
        if (verify_results(h_C_gpu, h_C_cpu, M, N, 1e-3f)) {
            printf("Results verified successfully!\n");
        } else {
            printf("Results verification failed!\n");
        }
    }
    
cleanup:
    // Cleanup
    destroyTimer(&timer);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
}

int main() {
    // Check CUDA device
    cudaError_t error = cuda_check_device();
    if (error != cudaSuccess) {
        printf("No suitable CUDA device found: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Initialize CUDA
    error = cuda_init();
    if (error != cudaSuccess) {
        printf("Failed to initialize CUDA: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Get and print device info
    CudaDeviceInfo info;
    error = cuda_get_device_info(&info);
    if (error == cudaSuccess) {
        printf("CUDA Device: %s\n", info.name);
        printf("Compute Capability: %d.%d\n", info.major, info.minor);
        printf("Total Memory: %.2f GB\n", info.total_memory / (1024.0 * 1024.0 * 1024.0));
        printf("Multiprocessors: %d\n", info.multi_processor_count);
        printf("Max Threads per Block: %d\n", info.max_threads_per_block);
    }
    
    // Test various matrix sizes
    test_matmul(256, 256, 256);
    test_matmul(512, 512, 512);
    test_matmul(1024, 1024, 1024);
    test_matmul(2048, 2048, 2048);
    test_matmul(4096, 4096, 4096);
    
    // Test non-square matrices
    test_matmul(1024, 2048, 512);
    test_matmul(2048, 1024, 1024);
    
    // Cleanup
    cuda_cleanup();
    
    return 0;
}