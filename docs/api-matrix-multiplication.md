# Matrix Multiplication Challenge API

## Overview

The Matrix Multiplication Challenge is a GPU-accelerated performance verification mechanism that tests a provider node's computational capabilities. This challenge is used by schedulers to verify that providers have the necessary hardware to handle AI inference workloads efficiently.

## Endpoint

### `POST /challenge`

Executes a challenge based on the provided type. For matrix multiplication, the challenge performs the operation C = A × B.

## Request Format

### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `X-Signature` | string | Yes | ECDSA signature of the request body |
| `X-Timestamp` | string | Yes | Unix timestamp of the request |
| `X-Address` | string | Yes | Ethereum address of the scheduler |
| `Content-Type` | string | Yes | Must be `application/json` |

### Body

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "payload": {
    "A": [[number]],
    "B": [[number]]
  }
}
```

#### Parameters

- **type** (string, required): Must be `"MATRIX_MULTIPLICATION"`
- **payload** (object, required):
  - **A** (2D array of numbers, required): First matrix for multiplication
  - **B** (2D array of numbers, required): Second matrix for multiplication

#### Matrix Constraints

- Matrices must be valid for multiplication: columns of A must equal rows of B
- Maximum dimension: 4096x4096 (configurable)
- Minimum dimension: 1x1
- Values must be finite numbers (no NaN or Inf)
- Empty matrices are not allowed

## Response Format

### Success Response (200 OK)

```json
{
  "type": "MATRIX_MULTIPLICATION",
  "result": {
    "C": [[number]],
    "computation_time_ms": number,
    "backend": string,
    "gpu_info": {
      "name": string,
      "memory_mb": number,
      "compute_capability": string
    }
  }
}
```

#### Fields

- **type** (string): Echo of the request type
- **result** (object):
  - **C** (2D array): Result matrix from A × B
  - **computation_time_ms** (number): Time taken for computation in milliseconds
  - **backend** (string): Backend used - `"cuda"` for GPU or `"cpu"` for CPU
  - **gpu_info** (object, optional): Present only when GPU is used
    - **name** (string): GPU model name
    - **memory_mb** (number): GPU memory in megabytes
    - **compute_capability** (string): CUDA compute capability

### Error Responses

#### 400 Bad Request

Invalid matrix dimensions or values:

```json
{
  "error": "invalid matrix dimensions: A columns (3) must equal B rows (2)"
}
```

#### 401 Unauthorized

Invalid signature or unregistered scheduler:

```json
{
  "error": "unauthorized: invalid signature"
}
```

#### 500 Internal Server Error

GPU failure or computation error:

```json
{
  "error": "GPU computation failed: out of memory"
}
```

## Examples

### Small Matrix Multiplication (CPU)

**Request:**
```bash
curl -X POST http://localhost:8090/challenge \
  -H "Content-Type: application/json" \
  -H "X-Signature: 0x..." \
  -H "X-Timestamp: 1703001234" \
  -H "X-Address: 0x..." \
  -d '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
      "A": [[1, 2], [3, 4]],
      "B": [[5, 6], [7, 8]]
    }
  }'
```

**Response:**
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "result": {
    "C": [[19, 22], [43, 50]],
    "computation_time_ms": 0.012,
    "backend": "cpu"
  }
}
```

### Large Matrix Multiplication (GPU)

**Request:**
```bash
curl -X POST http://localhost:8090/challenge \
  -H "Content-Type: application/json" \
  -H "X-Signature: 0x..." \
  -H "X-Timestamp: 1703001234" \
  -H "X-Address: 0x..." \
  -d '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
      "A": <1024x1024 matrix>,
      "B": <1024x1024 matrix>
    }
  }'
```

**Response:**
```json
{
  "type": "MATRIX_MULTIPLICATION",
  "result": {
    "C": <1024x1024 result matrix>,
    "computation_time_ms": 45.3,
    "backend": "cuda",
    "gpu_info": {
      "name": "NVIDIA GeForce RTX 3090",
      "memory_mb": 24576,
      "compute_capability": "8.6"
    }
  }
}
```

### Testing with Send Request Tool

The Function Node provides a helper tool for testing:

```bash
# Set scheduler private key
export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)

# Small matrix test
go run github.com/fxnlabs/function-node/cmd/send_request /challenge \
  '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'

# Large matrix test (creates random 256x256 matrices)
go run github.com/fxnlabs/function-node/cmd/send_request /challenge \
  "$(python3 -c 'import json, random; size=256; print(json.dumps({"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[random.random() for _ in range(size)] for _ in range(size)], "B": [[random.random() for _ in range(size)] for _ in range(size)]}}))')"
```

## Performance Expectations

### CPU vs GPU Performance

| Matrix Size | CPU Backend | GPU Backend | Speedup |
|-------------|-------------|-------------|---------|
| 32x32 | ~0.1ms | ~2ms | 0.05x (CPU faster due to overhead) |
| 64x64 | ~1ms | ~2ms | 0.5x |
| 128x128 | ~8ms | ~3ms | 2.7x |
| 256x256 | ~50ms | ~5ms | 10x |
| 512x512 | ~400ms | ~15ms | 27x |
| 1024x1024 | ~3000ms | ~50ms | 60x |
| 2048x2048 | ~25s | ~400ms | 62x |
| 4096x4096 | ~200s | ~3s | 67x |

### Backend Selection

The node automatically selects the optimal backend:
- **CPU**: Used for matrices smaller than 64x64 (configurable)
- **GPU**: Used for larger matrices when available
- **Fallback**: Automatically falls back to CPU if GPU fails

## Authentication Details

### Signature Generation

The signature must be generated using the scheduler's private key:

```javascript
const ethers = require('ethers');

function signRequest(privateKey, body, timestamp, address) {
  const payload = JSON.stringify(body) + timestamp + address;
  const hash = ethers.utils.keccak256(ethers.utils.toUtf8Bytes(payload));
  const signingKey = new ethers.utils.SigningKey(privateKey);
  const signature = signingKey.signDigest(hash);
  return ethers.utils.joinSignature(signature);
}
```

### Scheduler Registration

Only registered schedulers can send challenges. The node verifies scheduler registration through the on-chain SchedulerRegistry contract.

## Rate Limiting

- Maximum request size: 50MB (configurable)
- Maximum matrix dimension: 4096x4096 (configurable)
- Timeout: 60 seconds for computation
- Concurrent challenges: Limited by GPU memory

## Monitoring

### Metrics

The following Prometheus metrics are exposed:

```
# Challenge request counter
function_node_challenge_requests_total{type="MATRIX_MULTIPLICATION",status="success|failure"}

# Computation time histogram
function_node_challenge_duration_seconds{type="MATRIX_MULTIPLICATION",backend="cuda|cpu"}

# Matrix size distribution
function_node_challenge_matrix_size{type="MATRIX_MULTIPLICATION",dimension="rows|cols"}

# GPU utilization during challenge
function_node_gpu_utilization_percent{challenge="MATRIX_MULTIPLICATION"}
```

### Health Check

Verify GPU availability:

```bash
curl http://localhost:8090/health

# Response includes GPU status
{
  "status": "healthy",
  "gpu": {
    "available": true,
    "backend": "cuda",
    "device": "NVIDIA GeForce RTX 3090"
  }
}
```

## Implementation Notes

### GPU Memory Management

- Pre-allocates GPU memory for common matrix sizes
- Implements memory pooling to reduce allocation overhead
- Automatically garbage collects unused GPU memory
- Falls back to CPU when GPU memory is exhausted

### Precision

- Uses 32-bit floating-point (float32) for GPU computations
- Maintains numerical stability for large matrices
- Results may differ slightly between CPU and GPU due to floating-point arithmetic

### Security Considerations

- Validates all input matrices for finite values
- Implements timeouts to prevent DoS attacks
- Sanitizes error messages to prevent information leakage
- Rate limits requests per scheduler address