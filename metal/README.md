# Metal Backend

This directory contains the Apple Metal GPU backend implementation for the Function Node.

## Structure

```
metal/
├── include/          # C header files
│   └── metal_backend.h
├── src/              # Implementation files
│   ├── metal_backend.m   # Objective-C Metal backend
│   └── matmul.metal      # Metal shader kernels
├── lib/              # Compiled libraries (generated)
│   ├── libmetal_backend.a
│   ├── libmetal_backend.dylib
│   └── matmul.metallib
└── Makefile          # Build configuration
```

## Building

From the metal directory:
```bash
make all        # Build all libraries and shaders
make static     # Build static library only
make dynamic    # Build dynamic library only
make metallib   # Compile Metal shaders only
make clean      # Clean build artifacts
```

From the project root:
```bash
make metal      # Build entire project with Metal support
```

## Requirements

- macOS 11.0 or later
- Xcode Command Line Tools
- Metal-capable GPU (all modern Macs)

## Features

- Custom Metal kernels for matrix multiplication
- Metal Performance Shaders (MPS) integration
- Automatic kernel selection based on matrix size
- Buffer pooling for efficient memory management
- Support for Apple Silicon unified memory architecture

## Testing

Run Metal-specific tests:
```bash
make test-metal
```

Run Metal benchmarks:
```bash
make benchmark-metal
```