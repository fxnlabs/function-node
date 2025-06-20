# Apple Metal GPU Support

This document describes the Apple Metal GPU backend implementation for the Function Node.

## Overview

The Function Node now supports Apple Metal for GPU-accelerated matrix multiplication challenges. This enables providers with Apple hardware (both Apple Silicon and discrete GPUs) to participate in the network and respond to performance challenges.

## Features

- **Native Metal Implementation**: Direct integration with Metal compute shaders
- **Unified Memory Support**: Optimized for Apple Silicon's unified memory architecture
- **Automatic Detection**: Metal backend is automatically selected on supported macOS systems
- **High Performance**: Achieves 200+ GFLOPS on Apple M-series chips
- **Seamless Integration**: Works with existing challenge system without modifications

## Architecture

### Backend Implementation

The Metal backend (`internal/gpu/metal_backend.go`) implements the standard `GPUBackend` interface:

- `MatrixMultiply()`: Performs matrix multiplication using Metal compute kernels
- `GetDeviceInfo()`: Reports GPU family, memory, and capabilities
- `IsAvailable()`: Checks for Metal support
- `Initialize()`: Sets up Metal device and compute pipelines
- `Cleanup()`: Releases Metal resources

### Build System

Metal support is controlled by the `metal` build tag:

```bash
# Build with Metal support
go build -tags metal

# Build with both Metal and CUDA
go build -tags "metal cuda"
```

### Priority Order

When multiple backends are available, the selection priority is:
1. Metal (on macOS)
2. CUDA
3. CPU (fallback)

## Performance

Benchmark results on Apple M3 Max:

| Matrix Size | Performance (GFLOPS) | Time (ms) |
|------------|---------------------|-----------|
| 64×64      | 2.5                 | 0.21      |
| 128×128    | 17.6                | 0.24      |
| 256×256    | 104.6               | 0.32      |
| 512×512    | 315.4               | 0.85      |
| 1024×1024  | 604.1               | 3.56      |

## Building and Testing

### Prerequisites

- macOS 10.13+ (High Sierra or later)
- Xcode Command Line Tools
- Go 1.21+

### Build Commands

```bash
# Build with Metal support
make metal

# Run Metal tests
make test-metal

# Run Metal benchmarks
make benchmark-metal

# Check Metal availability
make check-metal
```

### Example Usage

See `examples/metal_gpu_demo.go` for a complete example of using the Metal backend.

## Integration with Challenge System

The matrix multiplication challenger automatically detects and uses the Metal backend when:

1. The `backend` parameter is set to `"gpu"` or `"auto"`
2. Metal is available on the system
3. The matrix size is large enough to benefit from GPU acceleration (typically ≥100×100)

Example challenge request:
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 512,
    "backend": "gpu"
  }
}
```

Response will include:
```json
{
  "backend": "metal",
  "deviceInfo": {
    "name": "Apple M3 Max",
    "computeCapability": "Apple7 (Unified Memory)",
    "memoryGB": 96
  },
  "computationTimeMs": 3.25,
  "gflops": 82.49,
  ...
}
```

## Technical Details

### Metal Shader

The implementation uses a simple but efficient compute kernel for matrix multiplication:

```metal
kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]],
                   constant uint& M [[buffer(3)]],
                   constant uint& N [[buffer(4)]],
                   constant uint& K [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}
```

### Memory Management

- Uses Metal's shared memory mode for efficient data transfer
- No explicit host-to-device copies needed on Apple Silicon
- Automatic memory management through ARC (Automatic Reference Counting)

### Thread Configuration

- Thread group size optimized for Metal's execution model
- Automatic calculation based on matrix dimensions
- Maximum threads per threadgroup queried from device

## Future Enhancements

1. **Tiled Matrix Multiplication**: Implement shared memory optimization for larger matrices
2. **Metal Performance Shaders**: Use MPS for additional operations
3. **Mixed Precision**: Support for float16/bfloat16 computation
4. **Multi-GPU Support**: Handle Mac Pro systems with multiple GPUs
5. **Performance Profiling**: Integration with Metal System Trace

## Troubleshooting

### Common Issues

1. **"Metal device not available"**: Ensure you're running on macOS with a supported GPU
2. **Build errors**: Make sure Xcode Command Line Tools are installed
3. **Performance issues**: Check that you're building with the `metal` tag

### Debug Information

Enable debug logging to see Metal backend initialization:
```bash
export LOG_LEVEL=debug
./fxn start
```

## References

- [Metal Programming Guide](https://developer.apple.com/metal/Metal-Programming-Guide.pdf)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)