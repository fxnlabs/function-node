# CUDA Development Environment

This document describes the CUDA development environment for the Function Node project.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- CUDA Toolkit 12.2 or compatible version
- NVIDIA driver compatible with CUDA 12.2
- [direnv](https://direnv.net/) for automatic environment variable management
- Docker with NVIDIA Container Toolkit (for containerized deployment)

## Building with CUDA Support

### Local Build

1. Ensure CUDA toolkit is installed:
   ```bash
   nvcc --version
   ```

2. Set up environment variables with direnv:
   ```bash
   # The project includes a .envrc file that configures CUDA paths
   direnv allow
   ```

3. Build with CUDA support:
   ```bash
   make cuda
   ```

4. Build without CUDA (CPU fallback):
   ```bash
   make cpu
   ```

### Docker Build

1. Build Docker image with CUDA support:
   ```bash
   make docker-cuda
   ```

2. Run with Docker Compose (CUDA enabled):
   ```bash
   make docker-run-cuda
   ```

## CUDA Implementation Details

The CUDA implementation provides GPU-accelerated matrix multiplication for challenge verification:

- **Location**: `cuda/` directory contains CUDA source files
- **Main files**:
  - `matmul.cu` - CUDA kernel implementation
  - `matmul.h` - C header for CUDA functions
  - `cuda_utils.cu` - Utility functions for CUDA operations

### Build System

The CUDA build is integrated into the main Makefile:

- `make cuda` - Builds the binary with CUDA support
- `make cuda-compile` - Compiles only the CUDA sources
- `make test` - Runs tests with CUDA support
- `make benchmark` - Runs performance benchmarks

### GPU Backend Architecture

The GPU backend uses a plugin architecture:

1. **GPU Manager** (`internal/gpu/manager.go`) - Manages GPU backend initialization
2. **CUDA Backend** (`internal/gpu/cuda_backend.go`) - CUDA-specific implementation
3. **CPU Backend** (`internal/gpu/cpu_backend.go`) - Fallback for non-GPU systems

### Performance Considerations

- GPU acceleration is used for matrices larger than 64x64
- Smaller matrices use CPU implementation due to overhead
- The system automatically falls back to CPU if GPU fails

## Testing

### Unit Tests
```bash
make test
```

### Benchmarks
```bash
make benchmark
```

### CUDA-specific Tests
```bash
cd cuda && make test
```

## Troubleshooting

### Check CUDA Installation
```bash
make check-cuda
```

### Environment Setup Issues

If you encounter library path issues, ensure direnv is properly configured:

```bash
# Check if direnv is working
direnv status

# If direnv isn't installed, install it:
# Ubuntu/Debian: sudo apt install direnv
# macOS: brew install direnv
# Then add to your shell config (.bashrc, .zshrc, etc.):
# eval "$(direnv hook bash)"  # for bash
# eval "$(direnv hook zsh)"   # for zsh
```

### Common Issues

1. **CUDA not found**: Ensure CUDA toolkit is installed and in PATH
2. **Library not found**: Use `direnv allow` to set up LD_LIBRARY_PATH, or manually run `ldconfig` after building CUDA libraries
3. **GPU not detected**: Check NVIDIA driver with `nvidia-smi`
4. **direnv not loading**: Ensure direnv hook is added to your shell configuration

## Docker GPU Support

The `docker-compose.cuda.yml` file includes:
- GPU device reservation
- NVIDIA runtime configuration
- CUDA environment variables

Requirements:
- NVIDIA Container Toolkit installed
- Docker configured with NVIDIA runtime

## Development Workflow

1. Modify CUDA code in `cuda/` directory
2. Run `make cuda` to rebuild
3. Test with `make test`
4. Benchmark with `make benchmark`

## Deployment

### Production Docker Image
```bash
docker build -f Dockerfile.cuda -t function-node:cuda .
```

### Environment Variables
- `NVIDIA_VISIBLE_DEVICES=all` - Enable all GPUs
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - Required capabilities