# GPU Backend Abstraction Layer

This directory contains the GPU backend abstraction layer for the Function Node, enabling high-performance matrix multiplication using either CUDA-enabled GPUs or CPU fallback.

## Architecture

The GPU backend follows a strategy pattern with the following components:

### Core Components

1. **`backend.go`** - Defines the `GPUBackend` interface and `DeviceInfo` struct
2. **`manager.go`** - Manages backend selection and lifecycle
3. **`cuda_backend.go`** - CUDA implementation (compiled with `cuda` build tag)
4. **`cpu_backend.go`** - CPU fallback implementation
5. **`utils.go`** - Utility functions for float32/float64 conversions

### Build Tags

- **`cuda`** - Enables CUDA support (requires CUDA toolkit)
- **`cpu_fallback` or `!cuda`** - Uses CPU-only implementation

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

### Without CUDA (CPU only)
```bash
go build
```

### With CUDA support
```bash
go build -tags cuda
```

Note: CUDA support requires:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Properly configured cgo environment

## Implementation Details

### CUDA Backend
- Uses cuBLAS for optimized matrix multiplication
- Handles GPU memory allocation/deallocation
- Provides device property queries
- Falls back to CPU if initialization fails

### CPU Backend
- Simple nested loop implementation
- Always available as fallback
- No external dependencies

### Manager
- Detects available backends at runtime
- Thread-safe backend access
- Handles initialization and cleanup
- Provides unified interface

## Testing

Run tests with:
```bash
# CPU backend tests
go test -v ./internal/gpu/... -tags "!cuda"

# CUDA backend tests (requires CUDA)
go test -v ./internal/gpu/... -tags cuda
```

## Performance Considerations

- CUDA backend is recommended for matrices larger than 100×100
- CPU backend is sufficient for smaller matrices
- The manager automatically selects the best available backend
- Matrix data is converted between float32 (GPU) and float64 (CPU) as needed