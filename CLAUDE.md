# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Function Node is the core software for providers on the Function Network, a P2P system for decentralized AI inference. It acts as a secure, observable proxy to OpenAI-compatible LLM endpoints, authenticating requests from network gateways and responding to health/performance challenges from schedulers.

## Common Development Commands

### Building and Running
```bash
# Build the CLI binary (CPU only)
go build github.com/fxnlabs/function-node/cmd/fxn

# Build with CUDA support (GPU acceleration)
make cuda

# Generate a new node key
go run github.com/fxnlabs/function-node/cmd/fxn account new

# Get node address from key
go run github.com/fxnlabs/function-node/cmd/fxn account get

# Start the node
go run github.com/fxnlabs/function-node/cmd/fxn start
```

### Testing
```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests with coverage
go test -cover ./...

# Run GPU-specific tests
go test -v ./internal/gpu ./cuda

# Run benchmarks (including GPU)
go test -bench=. ./internal/gpu -benchtime=10s

# Generate mocks (requires mockery v2.43.2)
go install github.com/vektra/mockery/v2@v2.43.2
mockery
```

### Testing Endpoints
```bash
# Test gateway requests (OpenAI proxy)
export PRIVATE_KEY=$(jq -r .private_key scripts/gateway_test_key.json)
go run github.com/fxnlabs/function-node/cmd/send_request /v1/chat/completions '{"model": "meta/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test challenge endpoints
export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{"type": "IDENTITY", "payload": {}}'

# Test matrix multiplication challenge (GPU)
go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'
```

### Docker
```bash
# Build Docker image (CPU only)
docker build -t function-node .

# Build Docker image with CUDA support
docker build -f Dockerfile.cuda -t function-node:cuda .

# Run with docker-compose (CPU)
docker-compose up

# Run with docker-compose (GPU)
docker-compose -f docker-compose.cuda.yml up
```

## Architecture Overview

The Function Node follows a modular architecture with clear separation of concerns:

### Core Components

1. **Authentication (`internal/auth/`)**
   - Verifies provider registration via ProviderRegistry smart contract
   - Authenticates gateway requests using signature verification (X-Signature header)
   - Prevents replay attacks with nonce caching and timestamp validation
   - Authenticates scheduler challenges to prevent spoofing

2. **OpenAI Proxy (`internal/openai/`)**
   - Proxies requests to configured model backends
   - Supports chat completions (streaming/non-streaming), completions, embeddings, and models endpoints
   - Model routing configured via `model_backend.yaml`
   - Each model can have different backend URLs and authentication methods

3. **Challenge System (`internal/challenge/`)**
   - Strategy pattern implementation for extensible challenge types
   - Current challenges:
     - Identity: Returns node identity with GPU stats
     - Matrix Multiplication: Verifies GPU performance (CUDA accelerated)
     - Endpoint Reachable: Basic health check
   - Central ChallengeHandler delegates to specific challengers
   - GPU backend system (`internal/gpu/`) provides CUDA acceleration

4. **GPU Acceleration (`internal/gpu/`, `cuda/`)**
   - Modular GPU backend architecture with CPU fallback
   - CUDA implementation for matrix multiplication
   - Automatic GPU detection and initialization
   - Performance monitoring and metrics

5. **Smart Contract Integration (`internal/registry/`, `internal/contracts/`)**
   - Generic CachedRegistry pattern for efficient blockchain queries
   - Asynchronous polling with configurable intervals
   - In-memory caching to reduce RPC calls
   - Separate registries for Gateway, Scheduler, and Provider data

6. **Observability (`internal/metrics/`, `internal/logger/`)**
   - Prometheus metrics for monitoring
   - Structured logging with Zap
   - Configurable verbosity levels

### Logging Convention

- **IMPORTANT**: Always use the Zap logger from `internal/logger/` for all logging in this codebase
- Do NOT use Go's standard library `log` or `log/slog` packages
- The project has a pre-configured Zap logger that should be used consistently throughout the codebase
- Example usage:
  ```go
  import "github.com/fxnlabs/function-node/internal/logger"
  
  logger.Info("Processing request", zap.String("id", requestID))
  logger.Error("Failed to process", zap.Error(err))
  ```

### Request Flow

1. Gateway/Scheduler → Function Node (authenticated request)
2. Authentication middleware verifies signature and registration
3. Request routed to appropriate handler (OpenAI proxy or Challenge)
4. For OpenAI: Request proxied to configured backend
5. Response returned with appropriate headers and metrics

### Configuration

- **`config.yaml`**: Main node configuration
  - Node keyfile path and listen port
  - Logger verbosity
  - Registry polling intervals and smart contract addresses
  - RPC provider URL
  - Nonce cache TTL
  - Proxy connection settings

- **`model_backend.yaml`**: Model routing configuration
  - Maps model names to backend URLs
  - Supports bearer token and API key authentication per model

### Key Design Patterns

1. **Dependency Injection**: Clean separation of concerns using interfaces
2. **Strategy Pattern**: Extensible challenge system
3. **Factory Pattern**: Registry creation with caching
4. **Middleware Pattern**: Authentication and logging middleware

## Testing Strategy

- Unit tests for all core components with mocked dependencies
- Integration tests for contract interactions
- Mock generation using mockery for interfaces
- Test fixtures in `fixtures/` directory for contract ABIs and test data

## Integration with Function Network

This node integrates with the broader Function Network ecosystem:
- Registers with on-chain ProviderRegistry
- Receives requests from registered Gateways
- Responds to challenges from Schedulers
- Can be deployed locally using function-localnet with Tilt