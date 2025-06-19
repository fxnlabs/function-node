# Matrix Multiplication Challenge Examples

The enhanced matrix multiplication challenge supports GPU performance testing with flexible payload options.

## Payload Structure

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 1024,           // Optional: generate random matrices of this size
    "A": [[...]],          // Optional: specific matrix A
    "B": [[...]],          // Optional: specific matrix B
    "backend": "auto",     // Optional: "gpu", "cpu", or "auto" (default)
    "precision": "float64" // Optional: data type precision (default: "float64")
  }
}
```

## Example Requests

### 1. Random Matrix Generation (Small)
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 512
  }
}
```

Response includes the result matrix and performance metrics:
```json
{
  "C": [[...]], 
  "computationTimeMs": 45.67,
  "backend": "gpu",
  "device": "NVIDIA GeForce RTX 4090",
  "matrixSize": 512,
  "flops": 268435456
}
```

### 2. Large Matrix Performance Test
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 2048,
    "backend": "gpu"
  }
}
```

Response omits the result matrix due to size:
```json
{
  "computationTimeMs": 234.56,
  "backend": "gpu",
  "device": "NVIDIA GeForce RTX 4090",
  "matrixSize": 2048,
  "flops": 17179869184
}
```

### 3. Specific Matrices Test
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "A": [[1, 2], [3, 4]],
    "B": [[5, 6], [7, 8]]
  }
}
```

Response:
```json
{
  "C": [[19, 22], [43, 50]],
  "computationTimeMs": 0.12,
  "backend": "cpu",
  "device": "CPU only",
  "matrixSize": 2,
  "flops": 16
}
```

### 4. Force CPU Backend
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 1024,
    "backend": "cpu"
  }
}
```

## Testing with curl

```bash
# Assuming you have the scheduler test key
export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)

# Test with random matrices
go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 1024
  }
}'

# Test with specific backend
go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 2048,
    "backend": "gpu"
  }
}'
```

## Performance Metrics

The challenge returns several useful metrics:

- **computationTimeMs**: Time taken for the matrix multiplication in milliseconds
- **backend**: Actually used backend ("cpu" or "gpu")
- **device**: Hardware device information
- **matrixSize**: Size of the square matrices
- **flops**: Theoretical floating-point operations (2 × n³ for n×n matrices)

## GPU Detection

The challenger automatically detects available GPUs:
- NVIDIA GPUs via `nvidia-smi`
- AMD GPUs via `rocm-smi` (Linux)
- Apple Silicon GPUs on macOS

If GPU is requested but not available, it falls back to CPU with a warning.

## Advanced Examples

### 5. GPU Stress Test
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 4096,
    "backend": "gpu"
  }
}
```

This creates 4096×4096 matrices requiring ~768MB of GPU memory and ~134 billion FLOPs.

### 6. Benchmark Different Sizes
```bash
# Create a benchmark script
cat > benchmark_gpu.sh << 'EOF'
#!/bin/bash
SIZES=(64 128 256 512 1024 2048 4096)
for size in "${SIZES[@]}"; do
  echo "Testing ${size}x${size} matrices..."
  time go run github.com/fxnlabs/function-node/cmd/send_request /challenge "{
    \"type\": \"MATRIX_MULTIPLICATION\",
    \"payload\": {
      \"size\": $size,
      \"backend\": \"gpu\"
    }
  }"
  echo "---"
done
EOF
chmod +x benchmark_gpu.sh
./benchmark_gpu.sh
```

### 7. Memory-Efficient Large Matrix
For systems with limited GPU memory:
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 2048,
    "backend": "auto"
  }
}
```
The auto backend will use GPU if available memory is sufficient.

### 8. Custom Matrix Validation
Test with known results:
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "A": [[1, 0], [0, 1]],  // Identity matrix
    "B": [[5, 6], [7, 8]]
  }
}
```
Expected result: C = B (identity property)

## Performance Optimization Tips

1. **Matrix Size Selection**
   - CPU optimal: < 64×64
   - GPU threshold: ≥ 64×64
   - GPU optimal: ≥ 256×256

2. **Memory Considerations**
   - Matrix memory = 2 × n² × 4 bytes (float32)
   - GPU needs 3× matrix memory (A, B, C)
   - Example: 4096×4096 needs ~768MB GPU memory

3. **Backend Selection Strategy**
   - Use "auto" for production (automatic optimization)
   - Use "gpu" for benchmarking GPU performance
   - Use "cpu" for baseline comparisons

## Monitoring GPU Performance

While running challenges, monitor GPU usage:
```bash
# In another terminal
watch -n 0.5 nvidia-smi

# Or for detailed metrics
nvidia-smi dmon -s u
```

## Expected Performance Results

| Matrix Size | CPU Time | GPU Time | Speedup | GPU Memory |
|-------------|----------|----------|---------|------------|
| 64×64       | ~1ms     | ~2ms     | 0.5×    | ~0.1MB     |
| 256×256     | ~50ms    | ~5ms     | 10×     | ~1.5MB     |
| 1024×1024   | ~3s      | ~50ms    | 60×     | ~24MB      |
| 2048×2048   | ~25s     | ~400ms   | 62×     | ~96MB      |
| 4096×4096   | ~200s    | ~3s      | 67×     | ~768MB     |

## Error Handling Examples

### GPU Out of Memory
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "size": 8192,  // Too large for most GPUs
    "backend": "gpu"
  }
}
```
Response: Automatic fallback to CPU with warning

### Invalid Matrix Dimensions
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "A": [[1, 2, 3]],
    "B": [[4], [5]]  // Incompatible dimensions
  }
}
```
Response: Error with dimension mismatch details