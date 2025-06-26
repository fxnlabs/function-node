# GPU Quick Reference Guide

## Pre-flight Checklist

- [ ] NVIDIA GPU with CUDA Compute Capability 5.0+
- [ ] NVIDIA Driver 525.60.13+
- [ ] CUDA Toolkit 12.2+
- [ ] `nvidia-smi` shows GPU
- [ ] Docker with NVIDIA Container Toolkit (for containerized deployment)

## Quick Commands

### Build and Run

```bash
# Build with GPU support
make cuda

# Run the node
./fxn start

# Build without GPU (CPU only)
make cpu
```

### Docker

```bash
# Build GPU-enabled image
docker build -f Dockerfile.cuda -t function-node:cuda .

# Run with GPU
docker-compose -f docker-compose.cuda.yml up
```

### Testing GPU

```bash
# Check GPU detection
nvidia-smi

# Test GPU build
go test -v ./internal/gpu -run TestGPUDetection

# Benchmark GPU performance
go test -bench=BenchmarkMatrixMultiplication ./internal/challenge/challengers

# Quick matrix multiplication test
export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
./scripts/test_matrix_challenge.sh
```

## Configuration

### config.yaml

```yaml
# GPU configuration section
gpu:
  enabled: true
  backend: "auto"  # auto, cuda, or cpu
  matrix_size_threshold: 64  # Use GPU for matrices >= this size
  max_matrix_size: 4096
```

### Environment Variables

```bash
# Docker/Kubernetes
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Debug mode
export CUDA_LAUNCH_BLOCKING=1
export FXN_GPU_DEBUG=1
```

## Performance Guide

| Matrix Size | Backend | Expected Time |
|-------------|---------|---------------|
| 32×32       | CPU     | <1ms          |
| 64×64       | CPU     | ~1ms          |
| 256×256     | GPU     | ~5ms          |
| 1024×1024   | GPU     | ~50ms         |
| 4096×4096   | GPU     | ~3s           |

## Troubleshooting

### GPU Not Detected

```bash
# Check driver
nvidia-smi

# Check CUDA
nvcc --version

# Check library path
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Rebuild
make clean && make cuda
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Reduce max matrix size in config.yaml
gpu:
  max_matrix_size: 2048  # Reduce from 4096
```

### Docker GPU Issues

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check container toolkit
nvidia-container-cli info
```

## Monitoring

### Metrics Endpoints

- Prometheus: `http://localhost:8090/metrics`
- Health: `http://localhost:8090/health`

### Key Metrics

```
# GPU utilization
function_node_gpu_utilization_percent

# Matrix multiplication performance
function_node_challenge_matrix_multiplication_duration_seconds
function_node_challenge_matrix_multiplication_ops_per_second

# GPU errors
function_node_gpu_errors_total
```

### Real-time GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor GPU metrics
nvidia-smi dmon -s u

# Check temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv --loop=1
```

## Best Practices

1. **Memory Management**
   - Keep 20% GPU memory free for system
   - Monitor for memory leaks with `nvidia-smi`

2. **Performance**
   - Use GPU for matrices ≥ 64×64
   - Batch operations when possible
   - Monitor temperature to prevent throttling

3. **Deployment**
   - Use exclusive process mode in production
   - Set up alerts for GPU errors
   - Plan maintenance windows for driver updates

4. **Security**
   - Validate matrix dimensions
   - Set computation timeouts
   - Rate limit challenge requests

## Common Matrix Sizes

```bash
# Small test (CPU)
'{"A": [[1,2],[3,4]], "B": [[5,6],[7,8]]}'

# Medium test (GPU threshold)
'{"size": 64}'

# Large test (GPU required)
'{"size": 1024}'

# Stress test
'{"size": 4096}'
```

## Support

- Full deployment guide: [matrix-multiplication-deployment.md](matrix-multiplication-deployment.md)
- API reference: [api-matrix-multiplication.md](api-matrix-multiplication.md)
- GPU backend docs: [../internal/gpu/README.md](../internal/gpu/README.md)