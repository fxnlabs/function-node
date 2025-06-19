# Metal GPU Backend Implementation

This document describes the Apple Metal GPU backend implementation for the Function Node, enabling GPU-accelerated matrix multiplication challenges on macOS systems.

## Features

### Dual Implementation Strategy

The Metal backend provides two matrix multiplication implementations:

1. **Custom Metal Kernels** - For small to medium matrices (<512×512 by default)
   - Simple kernel for very small matrices
   - Tiled kernel with shared memory optimization for medium matrices
   - Lower overhead for frequent small operations
   - Optimized thread group sizes for Apple GPUs

2. **Metal Performance Shaders (MPS)** - For large matrices (≥512×512 by default)
   - Leverages Apple's highly optimized BLAS implementation
   - Can achieve up to 7 TFLOPS on M1 Max
   - Automatic memory management and optimization
   - Best performance for large-scale computations

### Memory Optimization

- **Buffer Pooling**: Reuses Metal buffers to minimize allocation overhead
- **Unified Memory Architecture**: Zero-copy access on Apple Silicon
- **Automatic Memory Management**: Buffers are returned to pool after use

### Performance Features

- Automatic selection between custom kernels and MPS based on matrix size
- Configurable MPS threshold for performance tuning
- Support for all Apple GPU families (Apple1 through Apple9)
- Optimized for both Apple Silicon and Intel Macs with discrete GPUs

## Building

### Prerequisites

- macOS 11.0 or later
- Xcode Command Line Tools
- Go 1.19 or later

### Build Commands

```bash
# Build with Metal support
make metal

# Compile Metal shaders to metallib (optional, for better performance)
make metal-shaders

# Check Metal availability
make check-metal

# Run Metal-specific tests
make test-metal

# Run Metal benchmarks
make benchmark-metal
```

## Usage

The Metal backend is automatically selected when available on macOS systems. No code changes are required to use it.

### Configuring MPS Threshold

You can adjust when the backend switches from custom kernels to MPS:

```go
// In your initialization code
backend := gpu.NewMetalBackend(logger)
backend.SetMPSThreshold(256) // Use MPS for matrices ≥ 256×256
```

## Performance

Expected performance on different Apple Silicon chips:

| Chip | Matrix Size | Custom Kernel | MPS | 
|------|------------|---------------|-----|
| M1 | 512×512 | ~400 GFLOPS | ~600 GFLOPS |
| M1 | 1024×1024 | ~500 GFLOPS | ~800 GFLOPS |
| M1 Pro | 1024×1024 | ~600 GFLOPS | ~1200 GFLOPS |
| M2 Max | 1024×1024 | ~720 GFLOPS | ~1500 GFLOPS |
| M3 Max | 1024×1024 | ~850 GFLOPS | ~1800 GFLOPS |

## Testing

### Unit Tests

```bash
# Run all Metal backend tests
go test -tags metal -v ./internal/gpu/...

# Run specific test
go test -tags metal -v -run TestMetalBackend_MPSThreshold ./internal/gpu/
```

### Benchmarks

```bash
# Run performance benchmarks
go test -tags metal -bench=. ./internal/gpu/ -benchtime=10s

# Compare custom kernel vs MPS performance
go test -tags metal -bench=BenchmarkMetalBackend_MPSComparison ./internal/gpu/
```

## Architecture

### File Structure

```
internal/gpu/
├── metal_backend.go          # Main Metal backend implementation
├── matmul.metal             # Metal shader source code
├── matmul.metallib          # Compiled Metal shader (generated)
├── metal_backend_test.go    # Metal-specific tests
└── README_METAL.md          # This file
```

### C/Objective-C Bridge

The implementation uses CGO to bridge between Go and Objective-C/Metal:

1. **Initialization**: Creates Metal device, command queue, and compiles shaders
2. **Buffer Management**: Pool-based allocation for efficient memory reuse  
3. **Kernel Dispatch**: Automatic selection between custom kernels and MPS
4. **Error Handling**: Comprehensive error checking and reporting

### Metal Shaders

The `matmul.metal` file contains optimized kernels:

- `matmul_simple`: Basic matrix multiplication for small matrices
- `matmul_tiled`: Tiled algorithm with shared memory for better cache usage
- `matmul_tiled_optimized`: Vectorized implementation (future enhancement)
- `matrix_transpose`: Helper kernel for optimizing memory access patterns

## Troubleshooting

### Common Issues

1. **"Metal device not available"**
   - Ensure you're running on macOS 11.0 or later
   - Check that your Mac has a Metal-capable GPU

2. **Performance lower than expected**
   - Verify you're using the release build (`-tags metal`)
   - Check if thermal throttling is occurring
   - Try adjusting the MPS threshold

3. **Build failures**
   - Ensure Xcode Command Line Tools are installed: `xcode-select --install`
   - Verify CGO is enabled: `CGO_ENABLED=1`

### Debug Output

Enable debug logging to see which implementation is being used:

```bash
# Set log level to debug
export LOG_LEVEL=debug
./fxn start
```

## Future Enhancements

1. **Extended Operations**
   - Convolution operations for ML challenges
   - FFT operations for signal processing
   - Custom kernels for specific workloads

2. **Advanced Optimizations**
   - Vectorized kernels using float4 operations
   - Persistent kernel optimization
   - Graph-based computation for multiple operations

3. **Multi-GPU Support**
   - Detect and utilize multiple GPUs (Mac Pro)
   - Load balancing across GPUs

## Contributing

When contributing to the Metal backend:

1. Ensure all tests pass on your target hardware
2. Include benchmark results for performance changes
3. Document any new Metal-specific features or requirements
4. Test on both Apple Silicon and Intel Macs if possible