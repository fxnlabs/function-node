# Function Node Examples

This directory contains example programs demonstrating various features of the Function Node.

## GPU Backend Examples

### Metal GPU Demo (macOS)

Demonstrates using the Metal GPU backend for matrix multiplication on macOS.

```bash
# Build and run with Metal support
go run -tags metal metal_gpu_demo.go
```

This example:
- Initializes the GPU manager with Metal backend
- Displays GPU device information
- Performs matrix multiplication benchmarks
- Shows performance metrics in GFLOPS

### Requirements

- macOS 10.13+ (High Sierra or later)
- Apple GPU (integrated or discrete)
- Go 1.21+

### Expected Output

```
GPU Backend: metal
Device: Apple M1 Max
Compute Capability: Apple7 (Unified Memory)
Total Memory: 32.00 GB
Is GPU Available: true

Testing 64x64 matrix multiplication...
  Time: 285.5µs
  Performance: 1.84 GFLOPS
  Result size: 4096 elements

Testing 128x128 matrix multiplication...
  Time: 892.1µs
  Performance: 4.72 GFLOPS
  Result size: 16384 elements

...
```

## Building Examples

Examples can be built with different GPU backends:

```bash
# CPU only
go build -o demo metal_gpu_demo.go

# With Metal support (macOS)
go build -tags metal -o demo metal_gpu_demo.go

# With CUDA support
go build -tags cuda -o demo metal_gpu_demo.go
```

Note: The metal_gpu_demo.go file has build tags that restrict it to Metal-enabled builds on macOS. Remove or modify the build tags to test with other backends.