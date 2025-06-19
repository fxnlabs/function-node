# CUDA Matrix Multiplication Implementation

This directory contains an optimized CUDA implementation of matrix multiplication for the Function Node challenge system.

## Features

- **Tiled Matrix Multiplication**: Uses shared memory for optimal performance
- **Adaptive Kernel Selection**: Chooses between simple and tiled kernels based on matrix size
- **Configurable Tile Size**: Currently set to 32x32 for optimal performance on most GPUs
- **Boundary Checking**: Handles matrices of arbitrary sizes
- **Performance Metrics**: Includes utilities for measuring GFLOPS and bandwidth
- **CGO Integration**: Go wrapper for seamless integration with Function Node

## Performance Targets

The implementation meets the following performance requirements:
- 256x256: < 10ms
- 512x512: < 20ms  
- 1024x1024: < 100ms
- 2048x2048: < 500ms
- 4096x4096: < 2000ms

## Building

### Prerequisites
- CUDA Toolkit (>= 10.0)
- NVIDIA GPU with Compute Capability >= 3.0
- Go 1.19+ (for CGO wrapper)
- [direnv](https://direnv.net/) for automatic environment setup (recommended)

### Compilation

```bash
# First, set up the environment (recommended)
# The project root .envrc file automatically configures CUDA paths
cd .. && direnv allow && cd cuda

# Build the shared library and test program
make

# Build with debug symbols
make debug

# Run tests
make test

# Install library system-wide (requires sudo)
sudo make install
```

**Note**: The parent directory's `.envrc` file automatically sets the `LD_LIBRARY_PATH` to include the CUDA library path, eliminating the need for manual library path configuration.

## Usage

### From C/C++

```c
#include "matmul.h"

// Initialize CUDA
cuda_init();

// Check device
if (cuda_check_device() != cudaSuccess) {
    // Handle error
}

// Perform matrix multiplication
float *A, *B, *C;  // Your matrices
int M = 1024, N = 1024, K = 1024;
cudaError_t err = matmul_cuda(A, B, C, M, N, K);

// Cleanup
cuda_cleanup();
```

### From Go

```go
import "github.com/fxnlabs/function-node/cuda"

// Initialize CUDA
err := cuda.Init()
if err != nil {
    // Handle error
}
defer cuda.Cleanup()

// Create matrices
A := make([]float32, M*K)
B := make([]float32, K*N)

// Perform multiplication
C, err := cuda.MatMul(A, B, M, N, K)
if err != nil {
    // Handle error
}
```

## Files

- `matmul.cu` - Main CUDA kernel implementation
- `matmul.h` - C header file
- `cuda_utils.cu` - Utility functions for error checking and performance measurement
- `cuda_utils.h` - Utilities header
- `matmul_cgo.go` - Go CGO wrapper
- `matmul_test.go` - Go tests and benchmarks
- `test_matmul.cu` - Standalone C++ test program
- `Makefile` - Build configuration

## Optimization Details

1. **Shared Memory**: Each thread block loads tiles of matrices A and B into shared memory to reduce global memory accesses

2. **Coalesced Memory Access**: Threads access contiguous memory locations for optimal bandwidth utilization

3. **Loop Unrolling**: The inner computation loop is unrolled for better instruction-level parallelism

4. **Adaptive Kernel**: Small matrices use a simpler kernel to avoid overhead from tiling

5. **Optimal Block Size**: Uses 32x32 thread blocks to match warp size and maximize occupancy

## Performance Analysis

The implementation achieves high efficiency by:
- Minimizing global memory accesses through tiling
- Maximizing memory bandwidth through coalesced access patterns
- Achieving high arithmetic intensity in the inner loop
- Utilizing all available SMs through proper grid configuration

Expected performance on modern GPUs (e.g., RTX 3080):
- 1024x1024: ~3000 GFLOPS
- 2048x2048: ~4000 GFLOPS
- 4096x4096: ~5000 GFLOPS