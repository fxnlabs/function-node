#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// Utility function to get CUDA error string
extern "C" const char* cudaGetErrorStringExtended(cudaError_t error) {
    return cudaGetErrorString(error);
}

// Check and print CUDA error
void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, 
                cudaGetErrorString(error));
    }
}

// Allocate device memory with error checking
cudaError_t cudaMallocSafe(void** devPtr, size_t size) {
    cudaError_t error = cudaMalloc(devPtr, size);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate %zu bytes on device: %s\n", 
                size, cudaGetErrorString(error));
        *devPtr = nullptr;
    }
    return error;
}

// Allocate pinned host memory
cudaError_t cudaHostAllocSafe(void** ptr, size_t size, unsigned int flags) {
    cudaError_t error = cudaHostAlloc(ptr, size, flags);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate %zu bytes of pinned memory: %s\n",
                size, cudaGetErrorString(error));
        *ptr = nullptr;
    }
    return error;
}

// Get memory usage information
cudaError_t getMemoryInfo(size_t* free, size_t* total) {
    return cudaMemGetInfo(free, total);
}

// Print device properties in a formatted way
void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", 
                cudaGetErrorString(error));
        return;
    }
    
    printf("Device %d: %s\n", device, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dimensions: %d x %d x %d\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions: %d x %d x %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
}

// Get optimal block size for a kernel
cudaError_t getOptimalBlockSize(int* blockSize, int* minGridSize,
                               void* kernel, size_t dynamicSMemSize) {
    return cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, 
                                             kernel, dynamicSMemSize, 0);
}

// Create CUDA timer
cudaError_t createTimer(CudaTimer* timer) {
    cudaError_t error;
    
    error = cudaEventCreate(&timer->start);
    if (error != cudaSuccess) return error;
    
    error = cudaEventCreate(&timer->stop);
    if (error != cudaSuccess) {
        cudaEventDestroy(timer->start);
        return error;
    }
    
    timer->elapsed_ms = 0.0f;
    return cudaSuccess;
}

// Start timer
cudaError_t startTimer(CudaTimer* timer) {
    return cudaEventRecord(timer->start, 0);
}

// Stop timer
cudaError_t stopTimer(CudaTimer* timer) {
    cudaError_t error;
    
    error = cudaEventRecord(timer->stop, 0);
    if (error != cudaSuccess) return error;
    
    error = cudaEventSynchronize(timer->stop);
    if (error != cudaSuccess) return error;
    
    return cudaEventElapsedTime(&timer->elapsed_ms, timer->start, timer->stop);
}

// Destroy timer
void destroyTimer(CudaTimer* timer) {
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
}

// Memory bandwidth calculation helper
float calculateBandwidth(size_t bytes, float time_ms) {
    // Convert to GB/s
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

// FLOPS calculation helper for matrix multiplication
float calculateGFLOPS(int M, int N, int K, float time_ms) {
    // 2*M*N*K operations (multiply-add for each element)
    double flops = 2.0 * M * N * K;
    return (float)(flops / (time_ms / 1000.0) / 1e9);
}

// Check if device supports unified memory
bool supportsUnifiedMemory(int device) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    return prop.managedMemory != 0;
}

// Set device with error checking
cudaError_t setDeviceSafe(int device) {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        return error;
    }
    
    if (device >= device_count) {
        fprintf(stderr, "Invalid device ID %d. Only %d devices available.\n",
                device, device_count);
        return cudaErrorInvalidDevice;
    }
    
    return cudaSetDevice(device);
}