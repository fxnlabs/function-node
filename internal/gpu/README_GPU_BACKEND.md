# GPU Backend Abstraction Layer

This directory contains the GPU backend abstraction layer for the Function Node, enabling high-performance matrix multiplication using CUDA-enabled GPUs, Apple Metal GPUs, or CPU fallback.

## Architecture

The GPU backend follows a strategy pattern with the following components:

### Core Components

1. **`backend.go`** - Defines the `GPUBackend` interface and `DeviceInfo` struct
2. **`manager.go`** - Manages backend selection and lifecycle
3. **`cuda_backend.go`** - CUDA implementation (compiled with `cuda` build tag)
4. **`metal_backend.go`** - Metal implementation for macOS (compiled with `metal` build tag)
5. **`cpu_backend.go`** - CPU fallback implementation
6. **`utils.go`** - Utility functions for float32/float64 conversions

### Build Tags

- **`cuda`** - Enables CUDA support (requires CUDA toolkit)
- **`metal`** - Enables Metal support (macOS only)
- **No tags** - Uses CPU-only implementation

## Usage

```go
// Create a GPU manager
logger := slog.Default()
manager, err := gpu.NewManager(logger)
if err != nil {
    return err
}
defer manager.Cleanup()

// Get device information
info := manager.GetDeviceInfo()
fmt.Printf("Using %s backend on %s\n", manager.GetBackendType(), info.Name)

// Perform matrix multiplication
// C = A * B where A is m×k, B is k×n
result, err := manager.MatrixMultiply(a, b, m, k, n)
```

## Building

### Without GPU (CPU only)
```bash
go build
```

### With CUDA support
```bash
go build -tags cuda
```

### With Metal support (macOS)
```bash
go build -tags metal
```

### With both CUDA and Metal
```bash
go build -tags "cuda metal"
```

Note: 
- CUDA support requires:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit installed
  - Properly configured cgo environment
- Metal support requires:
  - macOS 10.13+ (High Sierra or later)
  - Apple GPU (integrated or discrete)

## Implementation Details

### CUDA Backend
- Uses optimized tiled matrix multiplication kernel
- Handles GPU memory allocation/deallocation
- Provides device property queries
- Falls back to CPU if initialization fails

### Metal Backend
- Uses Metal Shading Language compute kernels
- Leverages unified memory architecture on Apple Silicon
- Optimized for both discrete and integrated Apple GPUs
- Provides GPU family and memory information

### CPU Backend
- Uses Gonum for optimized BLAS operations
- Always available as fallback
- No external dependencies

### Manager
- Detects available backends at runtime
- Priority order: Metal (on macOS) → CUDA → CPU
- Thread-safe backend access
- Handles initialization and cleanup
- Provides unified interface

## Testing

Run tests with:
```bash
# CPU backend tests
go test -v ./internal/gpu/...

# CUDA backend tests (requires CUDA)
go test -v ./internal/gpu/... -tags cuda

# Metal backend tests (requires macOS)
go test -v ./internal/gpu/... -tags metal

# All backends (on macOS with CUDA)
go test -v ./internal/gpu/... -tags "cuda metal"
```

## Performance Considerations

- GPU backends (CUDA/Metal) are recommended for matrices larger than 100×100
- CPU backend is sufficient for smaller matrices
- The manager automatically selects the best available backend
- Matrix data is converted between float32 (GPU) and float64 (CPU) as needed
- Metal backend benefits from unified memory on Apple Silicon (no host-device transfer overhead)
- CUDA backend uses tiled multiplication for optimal memory access patterns