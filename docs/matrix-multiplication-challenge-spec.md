# Matrix Multiplication Challenge Feature Specification

## Executive Summary

This document specifies the implementation of a GPU-accelerated matrix multiplication challenge for the Function Network. The challenge verifies that provider nodes are running suitable GPU hardware by requiring them to perform matrix multiplication operations within performance thresholds. This ensures network participants have the computational resources necessary for AI inference workloads.

## Objectives

1. **Verify GPU Presence**: Ensure nodes have NVIDIA GPU hardware capable of accelerated computation
2. **Measure Performance**: Quantify computational capability through timed matrix operations
3. **Prevent Fraud**: Use cryptographic verification to ensure computations are performed correctly
4. **Future Extensibility**: Design architecture to support additional GPU backends (AMD ROCm, Apple Metal)

## Technical Requirements

### Core Requirements

- **Primary Target**: NVIDIA CUDA-capable GPUs (compute capability 3.5+)
- **Matrix Sizes**: Support configurable sizes from 256×256 to 8192×8192
- **Performance Threshold**: Complete 1024×1024 multiplication in <100ms on modern GPUs
- **Verification**: Implement Freivalds' algorithm for probabilistic result verification
- **Build System**: Docker-based development environment with CUDA toolkit
- **Language Integration**: Use cgo for Go-CUDA interoperability

### Development Requirements

- **CPU Fallback**: Hidden behind `cpu_fallback` build tag for development environments
- **Testing**: Comprehensive test suite with both GPU and CPU paths
- **Benchmarking**: Performance measurement tools for different matrix sizes
- **Documentation**: Clear setup and deployment instructions

## Architecture Design

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Function Node                         │
├─────────────────────────────────────────────────────────┤
│                 Challenge Handler                        │
│                        │                                 │
│     ┌──────────────────┴────────────────┐              │
│     │   MatrixMultiplicationChallenger   │              │
│     └──────────────────┬────────────────┘              │
│                        │                                 │
│     ┌──────────────────┴────────────────┐              │
│     │         GPU Backend Manager        │              │
│     │  ┌─────────┐ ┌─────────┐ ┌──────┐ │              │
│     │  │  CUDA   │ │  ROCm   │ │ Metal│ │              │
│     │  │ Backend │ │ Backend │ │Backend│ │              │
│     │  └─────────┘ └─────────┘ └──────┘ │              │
│     └────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **MatrixMultiplicationChallenger**: Main challenge implementation
2. **GPU Backend Manager**: Abstraction layer for different GPU implementations
3. **CUDA Backend**: NVIDIA GPU implementation using CUDA kernels
4. **Verification Module**: Implements Freivalds' algorithm
5. **Performance Monitor**: Tracks execution time and throughput

## API Specification

### Challenge Request

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 1024,              // Matrix dimensions (n×n)
    "matrixA": [[...]],        // Optional: specific matrix A
    "matrixB": [[...]],        // Optional: specific matrix B
    "seed": 12345,             // Optional: for reproducible random matrices
    "precision": "float32",    // Optional: float16, float32, float64
    "verificationSamples": 5   // Optional: number of Freivalds checks
  }
}
```

### Challenge Response

```json
{
  "success": true,
  "result": {
    "computationTimeMs": 45.67,
    "backend": "cuda",
    "device": {
      "name": "NVIDIA GeForce RTX 4090",
      "computeCapability": "8.9",
      "memoryGB": 24,
      "cudaCores": 16384
    },
    "performance": {
      "gflops": 47.89,
      "matrixSize": 1024,
      "precision": "float32"
    },
    "verification": {
      "merkleRoot": "0x1234567890abcdef...",
      "resultSamples": [
        {"row": 0, "col": 0, "value": 123.45},
        {"row": 511, "col": 511, "value": 678.90}
      ]
    }
  }
}
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

1. **CUDA Development Environment**
   - Create Dockerfile.cuda with NVIDIA base image
   - Set up build toolchain with nvcc
   - Configure cgo for CUDA integration

2. **Basic CUDA Kernel**
   - Implement naive matrix multiplication kernel
   - Create cgo wrapper functions
   - Test GPU memory allocation/deallocation

3. **Build System**
   - Add Makefile with CUDA compilation rules
   - Implement build tags for GPU/CPU modes
   - Create docker-compose.cuda.yml for development

### Phase 2: Integration (Week 3-4)

1. **Challenge Implementation**
   - Create MatrixMultiplicationChallenger
   - Integrate with existing challenge system
   - Implement request/response handling

2. **Performance Optimization**
   - Optimize CUDA kernel (tiling, shared memory)
   - Implement matrix size-based algorithm selection
   - Add performance monitoring

3. **Verification System**
   - Implement Freivalds' algorithm
   - Add Merkle tree generation for results
   - Create verification test suite

### Phase 3: Testing & Hardening (Week 5-6)

1. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests with mock scheduler
   - Performance benchmarks across GPU models

2. **Error Handling**
   - GPU availability detection
   - Graceful fallback mechanisms
   - Resource cleanup on errors

3. **Documentation**
   - Deployment guide
   - Performance tuning guide
   - Troubleshooting documentation

## CUDA Implementation Details

### Kernel Design

```cuda
// Optimized matrix multiplication kernel
__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int n, int tile_size
) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Tiled matrix multiplication
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < n && t * TILE_SIZE + tx < n)
            sA[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (col < n && t * TILE_SIZE + ty < n)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[ty][k] * sB[k][tx];
            
        __syncthreads();
    }
    
    // Write result
    if (row < n && col < n)
        C[row * n + col] = sum;
}
```

### CGO Integration

```go
// cuda_backend.go
// +build cuda

package gpu

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcuda
#cgo LDFLAGS: -L. -lmatmul_cuda

#include <cuda_runtime.h>
#include "matmul.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

type CUDABackend struct {
    deviceID int
    deviceProps C.cudaDeviceProp
}

func (c *CUDABackend) MatrixMultiply(a, b []float32, n int) ([]float32, error) {
    // Allocate GPU memory
    var d_a, d_b, d_c *C.float
    size := C.size_t(n * n * 4) // sizeof(float)
    
    if err := checkCudaError(C.cudaMalloc(&d_a, size)); err != nil {
        return nil, fmt.Errorf("failed to allocate GPU memory for A: %w", err)
    }
    defer C.cudaFree(d_a)
    
    // Copy data to GPU
    C.cudaMemcpy(d_a, unsafe.Pointer(&a[0]), size, C.cudaMemcpyHostToDevice)
    
    // Launch kernel
    C.matmul_cuda(d_a, d_b, d_c, C.int(n))
    
    // Copy result back
    result := make([]float32, n*n)
    C.cudaMemcpy(unsafe.Pointer(&result[0]), d_c, size, C.cudaMemcpyDeviceToHost)
    
    return result, nil
}
```

## Build Configuration

### Makefile

```makefile
# Build configuration for CUDA support
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -arch=sm_35 -O3 --ptxas-options=-v

# Build targets
.PHONY: all cuda cpu test

all: cuda

cuda:
	$(NVCC) $(NVCC_FLAGS) -c cuda/matmul.cu -o cuda/matmul.o
	$(NVCC) $(NVCC_FLAGS) -shared cuda/matmul.o -o libmatmul_cuda.so
	go build -tags cuda ./...

cpu:
	go build -tags cpu_fallback ./...

test:
	go test -tags "cuda test" ./...

benchmark:
	go test -bench=. -tags cuda ./...
```

### Docker Configuration

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Go
RUN apt-get update && apt-get install -y wget git
RUN wget https://go.dev/dl/go1.23.0.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Set up build environment
WORKDIR /app
COPY . .

# Build with CUDA support
RUN make cuda

# Runtime stage
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app
COPY --from=0 /app/fxn .
COPY --from=0 /app/*.so .

CMD ["./fxn", "start"]
```

## Testing Strategy

### Unit Tests

1. **Kernel Correctness**: Verify matrix multiplication results
2. **Performance Tests**: Ensure operations meet timing requirements
3. **Memory Management**: Test allocation/deallocation patterns
4. **Error Handling**: Test failure scenarios

### Integration Tests

```go
func TestMatrixMultiplicationChallenge(t *testing.T) {
    tests := []struct {
        name     string
        size     int
        maxTime  time.Duration
        backend  string
    }{
        {"Small Matrix GPU", 256, 10 * time.Millisecond, "cuda"},
        {"Medium Matrix GPU", 1024, 100 * time.Millisecond, "cuda"},
        {"Large Matrix GPU", 4096, 2 * time.Second, "cuda"},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

### Benchmarks

```go
func BenchmarkMatrixMultiplication(b *testing.B) {
    sizes := []int{256, 512, 1024, 2048, 4096}
    
    for _, size := range sizes {
        b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
            // Benchmark implementation
        })
    }
}
```

## Performance Targets

| Matrix Size | Target Time (ms) | GFLOPS |
|-------------|------------------|---------|
| 256×256     | < 5              | > 67    |
| 512×512     | < 20             | > 134   |
| 1024×1024   | < 100            | > 215   |
| 2048×2048   | < 500            | > 344   |
| 4096×4096   | < 3000           | > 458   |

## Security Considerations

1. **Resource Limits**: Prevent DoS through large matrix requests
2. **Memory Protection**: Validate matrix sizes before allocation
3. **Result Verification**: Cryptographic proof of computation
4. **Access Control**: Only registered schedulers can issue challenges

## Future Extensions

### Phase 4: Multi-GPU Backend Support

1. **AMD ROCm Backend**
   - Implement HIP kernels
   - Add ROCm build configuration
   - Create compatibility layer

2. **Apple Metal Backend**
   - Implement Metal Performance Shaders
   - Add macOS build support
   - Handle unified memory architecture

3. **Intel oneAPI Backend**
   - Support Intel GPUs
   - Implement SYCL kernels
   - Add Level Zero integration

### Advanced Features

1. **Batched Operations**: Multiple matrix multiplications in single kernel
2. **Mixed Precision**: FP16 for ML workloads, FP64 for scientific computing
3. **Sparse Matrix Support**: Optimized kernels for sparse matrices
4. **Tensor Operations**: Extend to higher-dimensional operations

## Deployment Guide

### Prerequisites

- NVIDIA GPU with CUDA capability 3.5+
- CUDA Toolkit 11.0+
- Docker with NVIDIA Container Toolkit
- Go 1.23+

### Quick Start

```bash
# Build CUDA-enabled container
docker build -f Dockerfile.cuda -t function-node:cuda .

# Run with GPU support
docker run --gpus all -v ./config.yaml:/app/config.yaml function-node:cuda

# Verify GPU detection
docker exec -it <container> ./fxn challenge test --type MATRIX_MULTIPLICATION
```

### Production Deployment

1. Ensure NVIDIA drivers are installed on host
2. Install NVIDIA Container Toolkit
3. Use provided docker-compose.cuda.yml
4. Monitor GPU utilization and temperature
5. Set appropriate resource limits

## Monitoring and Observability

### Metrics

- `challenge_matrix_mult_duration_ms`: Computation time histogram
- `challenge_matrix_mult_size`: Matrix size gauge
- `challenge_matrix_mult_gflops`: Performance counter
- `gpu_utilization_percent`: GPU usage percentage
- `gpu_memory_used_bytes`: GPU memory consumption

### Logging

```go
logger.Info("Matrix multiplication completed",
    zap.Int("size", size),
    zap.Float64("duration_ms", duration.Milliseconds()),
    zap.Float64("gflops", gflops),
    zap.String("device", deviceName),
)
```

## Conclusion

This specification provides a comprehensive approach to implementing GPU-accelerated matrix multiplication challenges for the Function Network. The design prioritizes performance verification while maintaining extensibility for future GPU platforms. By following this implementation plan, the network can ensure providers have the computational resources necessary for AI inference workloads.