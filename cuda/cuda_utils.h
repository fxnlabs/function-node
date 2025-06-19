#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <stdbool.h>

// Error checking macro
#define CUDA_SAFE_CALL(call) checkCudaError((call), __FILE__, __LINE__)

// Timer structure for performance measurement
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_ms;
} CudaTimer;

/**
 * Get extended CUDA error string
 */
const char* cudaGetErrorStringExtended(cudaError_t error);

/**
 * Check and print CUDA error
 */
void checkCudaError(cudaError_t error, const char* file, int line);

/**
 * Allocate device memory with error checking
 */
cudaError_t cudaMallocSafe(void** devPtr, size_t size);

/**
 * Allocate pinned host memory with error checking
 */
cudaError_t cudaHostAllocSafe(void** ptr, size_t size, unsigned int flags);

/**
 * Get current memory usage information
 */
cudaError_t getMemoryInfo(size_t* free, size_t* total);

/**
 * Print detailed device properties
 */
void printDeviceProperties(int device);

/**
 * Get optimal block size for a kernel
 */
cudaError_t getOptimalBlockSize(int* blockSize, int* minGridSize,
                               void* kernel, size_t dynamicSMemSize);

/**
 * Create a CUDA timer
 */
cudaError_t createTimer(CudaTimer* timer);

/**
 * Start timing
 */
cudaError_t startTimer(CudaTimer* timer);

/**
 * Stop timing and get elapsed time
 */
cudaError_t stopTimer(CudaTimer* timer);

/**
 * Destroy timer and free resources
 */
void destroyTimer(CudaTimer* timer);

/**
 * Calculate memory bandwidth in GB/s
 */
float calculateBandwidth(size_t bytes, float time_ms);

/**
 * Calculate GFLOPS for matrix multiplication
 */
float calculateGFLOPS(int M, int N, int K, float time_ms);

/**
 * Check if device supports unified memory
 */
bool supportsUnifiedMemory(int device);

/**
 * Set CUDA device with bounds checking
 */
cudaError_t setDeviceSafe(int device);

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H