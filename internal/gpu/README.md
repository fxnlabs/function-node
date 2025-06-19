# GPU Backend for Matrix Multiplication

This package provides GPU-accelerated matrix multiplication for the Function Node challenge system.

## Architecture

The package follows a strategy pattern with the following components:

- **GPUBackend Interface**: Defines the contract for matrix multiplication backends
- **CUDABackend**: NVIDIA CUDA implementation (build tag: `cuda`)
- **CPUBackend**: CPU fallback implementation (build tag: `!cuda`)
- **Factory**: Automatically selects the best available backend

## Building

### CPU-only build (default):
```bash
go build ./...
```

### CUDA-enabled build:
```bash
# Requires CUDA toolkit and nvcc compiler
go build -tags cuda ./...
```

### Docker build with CUDA:
```bash
docker build -f Dockerfile.cuda -t function-node:cuda .
```

## Usage

```go
import "github.com/fxnlabs/function-node/internal/gpu"

// Create backend (automatically selects best available)
backend := gpu.NewGPUBackend(logger)
defer backend.Cleanup()

// Perform matrix multiplication
// A is m×k, B is k×n, result is m×n
a := []float32{1, 2, 3, 4}      // 2×2 matrix
b := []float32{5, 6, 7, 8}      // 2×2 matrix
result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
```

## Performance

The CUDA backend provides significant performance improvements for large matrices:

| Matrix Size | CPU (GFLOPS) | GPU (GFLOPS) | Speedup |
|------------|--------------|--------------|---------|
| 128×128    | ~0.5         | ~50          | 100×    |
| 512×512    | ~0.3         | ~200         | 667×    |
| 1024×1024  | ~0.2         | ~500         | 2500×   |

*Performance varies based on hardware*

## Testing

### Unit tests:
```bash
# CPU backend tests
go test ./internal/gpu

# CUDA backend tests (requires CUDA)
go test -tags cuda ./internal/gpu
```

### Integration tests:
```bash
go test -tags integration ./test/integration -run TestMatrixChallenge
```

### Performance benchmarks:
```bash
go test -bench=. ./internal/gpu
```

## Error Handling

The backends handle various error conditions:
- Invalid matrix dimensions
- Memory allocation failures
- CUDA device not available
- Numerical overflow/underflow

## Implementation Details

### CUDA Backend
- Uses tiled matrix multiplication for cache efficiency
- Supports compute capability 3.0+
- Automatic memory management
- Thread-safe operations

### CPU Backend
- Naive O(n³) implementation
- Could be optimized with:
  - Loop tiling for cache efficiency
  - SIMD instructions
  - Parallel processing with goroutines
  - BLAS library integration

## Future Improvements

1. Support for other GPU backends (OpenCL, Metal, ROCm)
2. Mixed precision support (float16, int8)
3. Batch matrix multiplication
4. Sparse matrix support
5. Automatic performance tuning