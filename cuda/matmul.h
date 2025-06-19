#ifndef MATMUL_H
#define MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

// Structure to hold CUDA device information
typedef struct {
    char name[256];
    int major;
    int minor;
    int memory_clock_rate;      // in KHz
    int memory_bus_width;       // in bits
    size_t total_memory;        // in bytes
    size_t shared_memory_per_block;
    int max_threads_per_block;
    int max_threads_dim[3];
    int max_grid_size[3];
    int clock_rate;             // in KHz
    int multi_processor_count;
    int compute_capability;     // major * 10 + minor
} CudaDeviceInfo;

/**
 * Perform matrix multiplication C = A * B using CUDA
 * 
 * @param h_A Input matrix A (M x K) in row-major order
 * @param h_B Input matrix B (K x N) in row-major order
 * @param h_C Output matrix C (M x N) in row-major order
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @return cudaError_t CUDA error code (cudaSuccess on success)
 */
cudaError_t matmul_cuda(const float* h_A, const float* h_B, float* h_C,
                       int M, int N, int K);

/**
 * Get information about the current CUDA device
 * 
 * @param info Pointer to CudaDeviceInfo structure to fill
 * @return cudaError_t CUDA error code (cudaSuccess on success)
 */
cudaError_t cuda_get_device_info(CudaDeviceInfo* info);

/**
 * Check if a CUDA device is available and meets requirements
 * 
 * @return cudaError_t CUDA error code (cudaSuccess if device is available)
 */
cudaError_t cuda_check_device();

/**
 * Initialize CUDA context
 * 
 * @return cudaError_t CUDA error code (cudaSuccess on success)
 */
cudaError_t cuda_init();

/**
 * Cleanup CUDA resources
 * 
 * @return cudaError_t CUDA error code (cudaSuccess on success)
 */
cudaError_t cuda_cleanup();

#ifdef __cplusplus
}
#endif

#endif // MATMUL_H