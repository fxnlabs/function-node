# Function Node Metrics Guide

## Matrix Multiplication Challenge Metrics

The Function Node exposes Prometheus metrics for monitoring the performance and behavior of matrix multiplication challenges. These metrics are available at the `/metrics` endpoint.

### Available Metrics

#### Performance Metrics

- **`challenge_matrix_mult_duration_ms`** (Histogram)
  - Description: Duration of matrix multiplication computation in milliseconds
  - Buckets: Exponential from 1ms to ~32s
  - Use case: Monitor computation performance and identify outliers

- **`challenge_matrix_mult_gflops`** (Gauge)
  - Description: Performance of the last matrix multiplication in GFLOPS (Giga Floating-point Operations Per Second)
  - Use case: Track computational throughput

- **`challenge_matrix_mult_size`** (Gauge)
  - Description: Size of the matrix used in the last multiplication challenge
  - Use case: Correlate performance with problem size

#### Backend Metrics

- **`challenge_matrix_mult_backend_total`** (Counter)
  - Description: Total number of matrix multiplication challenges by backend
  - Labels: `backend` (cpu, cuda, opencl, etc.)
  - Use case: Track backend usage distribution

#### GPU Metrics

- **`gpu_utilization_percent`** (Gauge)
  - Description: Current GPU utilization percentage (0-100)
  - Note: Currently set to 0 if GPU monitoring is not available
  - Use case: Monitor GPU load

- **`gpu_memory_used_bytes`** (Gauge)
  - Description: GPU memory currently in use in bytes
  - Use case: Track GPU memory consumption

### Example Prometheus Queries

```promql
# 99th percentile computation time over the last 5 minutes
histogram_quantile(0.99, sum(rate(challenge_matrix_mult_duration_ms_bucket[5m])) by (le))

# Average GFLOPS performance
avg_over_time(challenge_matrix_mult_gflops[5m])

# Challenge rate by backend
sum by (backend) (rate(challenge_matrix_mult_backend_total[5m]))

# GPU memory usage in GB
gpu_memory_used_bytes / 1024 / 1024 / 1024
```

### Grafana Dashboard

A pre-configured Grafana dashboard is available at `docs/grafana-dashboard-matrix-multiplication.json`. To use it:

1. Import the JSON file into your Grafana instance
2. Select your Prometheus data source
3. The dashboard includes:
   - Computation duration percentiles (p50, p95, p99)
   - Current GFLOPS performance gauge
   - Backend usage distribution pie chart
   - Matrix size over time
   - GPU metrics (utilization and memory)
   - Challenge rate by backend

### Logging

The matrix multiplication challenger includes structured logging with zap:

- **Info level**: Challenge start/completion, backend selection, matrix generation
- **Debug level**: GPU initialization attempts, computation timing details
- **Error level**: Dimension mismatches, GPU failures, invalid payloads
- **Warn level**: GPU fallback scenarios

Example log entries:
```
INFO Starting matrix multiplication challenge
INFO Generated random matrices {"size": 1024, "backend": "auto", "precision": "float64"}
INFO GPU backend initialized successfully {"device_name": "NVIDIA GeForce RTX 3090", "compute_capability": "8.6", "memory_gb": 24.0}
INFO Matrix multiplication completed {"computation_time_ms": 123.45, "backend": "cuda", "matrix_size": 1024, "gflops": 17.34, "merkle_root": "0x1234..."}
```

### Integration with Monitoring Stack

To integrate with your monitoring stack:

1. Configure Prometheus to scrape the Function Node metrics endpoint:
   ```yaml
   scrape_configs:
     - job_name: 'function-node'
       static_configs:
         - targets: ['localhost:8080']  # Adjust port as needed
   ```

2. Set up alerts based on performance thresholds:
   ```yaml
   groups:
     - name: function_node_alerts
       rules:
         - alert: SlowMatrixMultiplication
           expr: histogram_quantile(0.99, challenge_matrix_mult_duration_ms_bucket) > 10000
           for: 5m
           annotations:
             summary: "Matrix multiplication is taking too long"
   ```

3. Use the metrics for capacity planning and optimization decisions based on backend performance and GPU utilization.