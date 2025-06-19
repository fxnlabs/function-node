#include "matmul.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Tile size for shared memory optimization
// This value is chosen to maximize occupancy on most NVIDIA GPUs
// 32x32 = 1024 threads per block, which is the maximum for many GPUs
// Each tile uses 32*32*4 = 4KB of shared memory per matrix
#define TILE_SIZE 32

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        return error; \
    } \
} while (0)

// Optimized tiled matrix multiplication kernel using shared memory
// This kernel implements the classic tiled matrix multiplication algorithm
// to maximize memory bandwidth utilization and minimize global memory accesses.
//
// Algorithm overview:
// 1. Divide matrices into TILE_SIZE x TILE_SIZE tiles
// 2. Each thread block computes one tile of the output matrix C
// 3. Load tiles from A and B into shared memory
// 4. Compute partial products using shared memory data
// 5. Accumulate results across all tiles
//
// Performance characteristics:
// - Reduces global memory accesses from O(MNK) to O(MNK/TILE_SIZE)
// - Achieves near-peak performance for large matrices
// - Shared memory bank conflicts are minimized with proper tile size
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float Cvalue = 0.0f;
    
    // Loop over tiles along the K dimension
    // Each iteration loads one tile from A and B, computes partial product
    // The loop ceiling division ensures we process all elements even if K is not divisible by TILE_SIZE
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from matrix A
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from matrix B
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize threads to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial product for this tile
        // The pragma unroll directive tells the compiler to unroll this loop
        // This reduces loop overhead and enables better instruction-level parallelism
        // Each thread computes one element of the output tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// Simple kernel for small matrices without shared memory optimization
// This kernel is more efficient for small matrices where:
// - The overhead of shared memory synchronization exceeds the benefit
// - The entire problem fits in L1/L2 cache
// - Matrix dimensions are not aligned to tile boundaries
//
// Each thread computes one element of C directly from global memory
__global__ void matmul_simple_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Main matrix multiplication function
extern "C" cudaError_t matmul_cuda(const float* h_A, const float* h_B, float* h_C,
                                  int M, int N, int K) {
    // Handle edge cases
    if (M <= 0 || N <= 0 || K <= 0) {
        // For zero-sized matrices, just return success without doing anything
        return cudaSuccess;
    }
    
    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Allocate device memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockDim, gridDim;
    
    // Choose kernel based on matrix size
    if (M <= 512 && N <= 512 && K <= 512) {
        // For small matrices, use simple kernel
        blockDim = dim3(16, 16);
        gridDim = dim3((N + blockDim.x - 1) / blockDim.x,
                      (M + blockDim.y - 1) / blockDim.y);
        matmul_simple_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // For larger matrices, use tiled kernel
        blockDim = dim3(TILE_SIZE, TILE_SIZE);
        gridDim = dim3((N + TILE_SIZE - 1) / TILE_SIZE,
                      (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return cudaSuccess;
}

// Get CUDA device information
extern "C" cudaError_t cuda_get_device_info(CudaDeviceInfo* info) {
    if (!info) {
        return cudaErrorInvalidValue;
    }
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Fill device info
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    
    info->major = prop.major;
    info->minor = prop.minor;
    info->memory_clock_rate = prop.memoryClockRate;
    info->memory_bus_width = prop.memoryBusWidth;
    info->total_memory = prop.totalGlobalMem;
    info->shared_memory_per_block = prop.sharedMemPerBlock;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->max_threads_dim[0] = prop.maxThreadsDim[0];
    info->max_threads_dim[1] = prop.maxThreadsDim[1];
    info->max_threads_dim[2] = prop.maxThreadsDim[2];
    info->max_grid_size[0] = prop.maxGridSize[0];
    info->max_grid_size[1] = prop.maxGridSize[1];
    info->max_grid_size[2] = prop.maxGridSize[2];
    info->clock_rate = prop.clockRate;
    info->multi_processor_count = prop.multiProcessorCount;
    info->compute_capability = prop.major * 10 + prop.minor;
    
    return cudaSuccess;
}

// Check if CUDA device is available
extern "C" cudaError_t cuda_check_device() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        return cudaErrorNoDevice;
    }
    
    // Check if current device supports our requirements
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Require at least compute capability 3.0 for good performance
    if (prop.major < 3) {
        fprintf(stderr, "Device compute capability %d.%d is too low. Minimum required: 3.0\n",
                prop.major, prop.minor);
        return cudaErrorInsufficientDriver;
    }
    
    return cudaSuccess;
}

// Initialize CUDA context
extern "C" cudaError_t cuda_init() {
    // This will initialize the CUDA context
    CUDA_CHECK(cudaFree(0));
    return cudaSuccess;
}

// Cleanup CUDA resources
extern "C" cudaError_t cuda_cleanup() {
    CUDA_CHECK(cudaDeviceReset());
    return cudaSuccess;
}