# Matrix Multiplication Challenge Deployment Guide

This guide covers deployment and configuration of the Function Node with GPU support for the matrix multiplication challenge.

## Table of Contents

- [Quick Start](#quick-start)
- [GPU Requirements](#gpu-requirements)
- [Installation](#installation)
- [Docker Deployment](#docker-deployment)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Quick Start

The fastest way to get started with GPU support:

```bash
# Check GPU availability
nvidia-smi

# Set up environment variables automatically (recommended)
# The project includes a .envrc file that sets LD_LIBRARY_PATH for CUDA
direnv allow

# Build with CUDA support
make cuda

# Run the node
./fxn start

# Test the matrix multiplication challenge
export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
go run github.com/fxnlabs/function-node/cmd/send_request /challenge \
  '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'
```

## GPU Requirements

### Hardware Requirements

- **NVIDIA GPU**: CUDA Compute Capability 5.0 or higher
  - GeForce GTX 900 series or newer
  - Tesla K40 or newer
  - Quadro M2000 or newer
- **Memory**: At least 4GB GPU memory for production workloads
- **Driver**: NVIDIA driver version 525.60.13 or higher

### Software Requirements

- **CUDA Toolkit**: 12.2 or compatible version
- **cuBLAS**: Included with CUDA Toolkit
- **Go**: 1.21 or higher with CGO enabled
- **GCC**: Compatible with your CUDA version
- **direnv**: Recommended for automatic environment setup

### Verifying GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check cuBLAS
find /usr/local/cuda -name "libcublas.so*"

# Run GPU detection test
go test -v ./internal/gpu -run TestGPUDetection
```

## Installation

### Ubuntu/Debian

```bash
# Install NVIDIA driver
sudo apt update
sudo apt install nvidia-driver-525

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-12-2

# Set environment variables (or use direnv for automatic setup)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Alternative: Use direnv for automatic environment management
# Install direnv: sudo apt install direnv
# Add to ~/.bashrc: eval "$(direnv hook bash)"
# Then in the project directory: direnv allow

# Build Function Node with CUDA
make cuda
```

### CentOS/RHEL

```bash
# Install NVIDIA driver
sudo yum install nvidia-driver-latest-dkms

# Install CUDA Toolkit
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo yum install cuda-12-2

# Set environment variables (or use direnv for automatic setup)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bash_profile
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bash_profile
source ~/.bash_profile

# Alternative: Use direnv for automatic environment management
# Install direnv: sudo yum install direnv
# Add to ~/.bash_profile: eval "$(direnv hook bash)"
# Then in the project directory: direnv allow

# Build Function Node with CUDA
make cuda
```

## Docker Deployment

### Prerequisites

1. **Install NVIDIA Container Toolkit**:

   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt update
   sudo apt install nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Verify Docker GPU support**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

### Building the Docker Image

```bash
# Build CUDA-enabled image
docker build -f Dockerfile.cuda -t function-node:cuda .

# Build with specific CUDA version
docker build -f Dockerfile.cuda \
  --build-arg CUDA_VERSION=12.2.0 \
  -t function-node:cuda-12.2 .
```

### Running with Docker Compose

1. **Configure `docker-compose.cuda.yml`**:

   ```yaml
   version: "3.8"
   services:
     function-node:
       image: function-node:cuda
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
       environment:
         - NVIDIA_VISIBLE_DEVICES=all
         - NVIDIA_DRIVER_CAPABILITIES=compute,utility
       volumes:
         - ./config.yaml:/app/config.yaml
         - ./model_backend.yaml:/app/model_backend.yaml
         - ./keyfile.json:/app/keyfile.json
       ports:
         - "8090:8090"
   ```

2. **Start the service**:
   ```bash
   docker-compose -f docker-compose.cuda.yml up -d
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: function-node
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: function-node
          image: function-node:cuda
          resources:
            limits:
              nvidia.com/gpu: 1
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "all"
            - name: NVIDIA_DRIVER_CAPABILITIES
              value: "compute,utility"
```

## Performance Tuning

### GPU Configuration

1. **Memory Management**:

   ```bash
   # Set GPU memory growth (prevents OOM)
   export CUDA_LAUNCH_BLOCKING=1
   export CUDA_DEVICE_MAX_CONNECTIONS=1
   ```

2. **CUDA Streams**:
   The implementation uses CUDA streams for concurrent operations. Tune the number of streams based on your GPU:

   ```go
   // In internal/gpu/cuda_backend.go
   const MAX_STREAMS = 4 // Adjust based on GPU capabilities
   ```

3. **Matrix Size Threshold**:
   Adjust the CPU/GPU crossover point in the configuration:
   ```yaml
   # config.yaml
   gpu:
     matrix_size_threshold: 64 # Use GPU for matrices larger than 64x64
     max_matrix_size: 4096 # Maximum allowed matrix dimension
   ```

### Benchmarking

Run performance benchmarks to tune your deployment:

```bash
# Run GPU benchmarks
go test -bench=. ./internal/gpu -benchtime=10s

# Run matrix multiplication benchmarks
go test -bench=BenchmarkMatrixMultiplication ./internal/challenge/challengers

# Profile GPU utilization
nvidia-smi dmon -s u -d 10
```

### Performance Metrics

Expected performance for matrix multiplication:

| Matrix Size | CPU Time | GPU Time | Speedup |
| ----------- | -------- | -------- | ------- |
| 64x64       | 1ms      | 2ms      | 0.5x    |
| 256x256     | 50ms     | 5ms      | 10x     |
| 1024x1024   | 3000ms   | 50ms     | 60x     |
| 4096x4096   | 200s     | 1s       | 200x    |

## Monitoring

### Prometheus Metrics

The node exposes GPU-specific metrics:

```yaml
# GPU utilization percentage
function_node_gpu_utilization_percent

# GPU memory usage
function_node_gpu_memory_used_bytes
function_node_gpu_memory_total_bytes

# Matrix multiplication performance
function_node_challenge_matrix_multiplication_duration_seconds
function_node_challenge_matrix_multiplication_ops_per_second

# GPU errors
function_node_gpu_errors_total
```

### Grafana Dashboard

Import the provided dashboard for GPU monitoring:

```json
{
  "dashboard": {
    "title": "Function Node GPU Metrics",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "function_node_gpu_utilization_percent"
          }
        ]
      },
      {
        "title": "Matrix Multiplication Performance",
        "targets": [
          {
            "expr": "rate(function_node_challenge_matrix_multiplication_ops_per_second[5m])"
          }
        ]
      }
    ]
  }
}
```

### Logging

Enable detailed GPU logging:

```yaml
# config.yaml
logger:
  level: debug
  gpu_debug: true
```

GPU-related log entries:

```
INFO  gpu/manager.go:45  GPU backend initialized  {"backend": "cuda", "device": "NVIDIA GeForce RTX 3090", "memory": "24GB"}
DEBUG gpu/cuda_backend.go:123  Matrix multiplication  {"size": "1024x1024", "duration": "45ms", "gflops": "47.2"}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found

**Error**: `cuda: library not found`

**Solution**:

```bash
# Option 1: Use direnv (recommended)
direnv allow  # This automatically sets LD_LIBRARY_PATH from .envrc

# Option 2: Manual setup
# Update library cache
sudo ldconfig /usr/local/cuda/lib64

# Verify CUDA libraries
ldd ./fxn | grep cuda

# Set LD_LIBRARY_PATH manually
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. GPU Out of Memory

**Error**: `CUDA error: out of memory`

**Solution**:

```bash
# Check GPU memory usage
nvidia-smi

# Reduce matrix size threshold
# In config.yaml:
gpu:
  max_matrix_size: 2048  # Reduce from 4096
```

#### 3. GPU Not Detected

**Error**: `No CUDA-capable device is detected`

**Solution**:

```bash
# Check driver installation
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify GPU permissions
ls -la /dev/nvidia*

# For Docker, ensure nvidia-container-toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### 4. Performance Degradation

**Symptoms**: Slower than expected GPU performance

**Solutions**:

1. **Check GPU throttling**:

   ```bash
   nvidia-smi -q -d PERFORMANCE
   ```

2. **Monitor temperature**:

   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
   ```

3. **Adjust power settings**:
   ```bash
   sudo nvidia-smi -pm 1  # Enable persistence mode
   sudo nvidia-smi -pl 300  # Set power limit (watts)
   ```

### Debug Mode

Enable comprehensive debugging:

```bash
# Set debug environment variables
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_DEBUG=1
export FXN_GPU_DEBUG=1

# Run with verbose logging
./fxn start --log-level=debug
```

### Health Checks

Implement health checks for GPU functionality:

```bash
# Test CUDA functionality
go test -v ./cuda -run TestCUDAHealth

# Benchmark specific operations
go test -bench=BenchmarkCuBLAS ./internal/gpu

# Stress test
./scripts/gpu_stress_test.sh
```

## API Reference

### Matrix Multiplication Challenge Endpoint

**Endpoint**: `POST /challenge`

**Headers**:

- `X-Signature`: Request signature
- `X-Timestamp`: Request timestamp
- `X-Address`: Scheduler address

**Request Body**:

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "A": [
      [1.0, 2.0],
      [3.0, 4.0]
    ],
    "B": [
      [5.0, 6.0],
      [7.0, 8.0]
    ]
  }
}
```

**Response**:

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "result": {
    "C": [
      [19.0, 22.0],
      [43.0, 50.0]
    ],
    "computation_time_ms": 0.523,
    "backend": "cuda",
    "gpu_info": {
      "name": "NVIDIA GeForce RTX 3090",
      "memory_mb": 24576,
      "compute_capability": "8.6"
    }
  }
}
```

### Performance Metrics Endpoint

**Endpoint**: `GET /metrics`

**GPU-specific metrics**:

```
# HELP function_node_gpu_utilization_percent Current GPU utilization
# TYPE function_node_gpu_utilization_percent gauge
function_node_gpu_utilization_percent{gpu="0",name="NVIDIA GeForce RTX 3090"} 85

# HELP function_node_challenge_matrix_multiplication_duration_seconds Time spent in matrix multiplication
# TYPE function_node_challenge_matrix_multiplication_duration_seconds histogram
function_node_challenge_matrix_multiplication_duration_seconds_bucket{le="0.001"} 100
function_node_challenge_matrix_multiplication_duration_seconds_bucket{le="0.01"} 500
function_node_challenge_matrix_multiplication_duration_seconds_bucket{le="0.1"} 900
function_node_challenge_matrix_multiplication_duration_seconds_bucket{le="1"} 950
function_node_challenge_matrix_multiplication_duration_seconds_bucket{le="+Inf"} 1000
```

### Configuration Reference

```yaml
# config.yaml GPU section
gpu:
  enabled: true
  backend: "cuda" # Options: cuda, cpu, auto
  matrix_size_threshold: 64
  max_matrix_size: 4096
  cuda:
    device_id: 0 # GPU device to use
    memory_fraction: 0.8 # Fraction of GPU memory to use
    streams: 4 # Number of CUDA streams
  monitoring:
    enabled: true
    interval: 10s
```

## Best Practices

1. **Resource Allocation**:

   - Reserve 20% of GPU memory for system overhead
   - Use exclusive GPU mode for production deployments
   - Monitor GPU temperature and throttling

2. **Error Handling**:

   - Always implement CPU fallback
   - Log GPU errors with context
   - Implement circuit breakers for GPU failures

3. **Security**:

   - Validate matrix dimensions before GPU operations
   - Implement timeouts for long-running computations
   - Sanitize error messages to avoid information leakage

4. **Deployment**:
   - Use GPU node labels in Kubernetes
   - Implement proper health checks
   - Monitor GPU metrics continuously
   - Plan for GPU maintenance windows
