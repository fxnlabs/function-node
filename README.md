# Function Node

The Function Node is the core software for providers on the Function Network, a peer-to-peer system for decentralized AI inference. It serves as a secure, observable proxy to any OpenAI-compatible LLM endpoint, authenticating requests from network gateways and responding to health and performance challenges from schedulers. The node interacts with on-chain registries to stay synced with the network.

## Key Features

- **OpenAI-Compatible Proxy**: Supports chat completions, completions, embeddings, and models endpoints
- **Authentication**: Cryptographic signature verification for gateway and scheduler requests
- **GPU Challenges**: Performance verification through GPU-accelerated matrix multiplication
- **Registry Integration**: Real-time synchronization with on-chain provider, gateway, and scheduler registries
- **Model Routing**: Flexible backend configuration for different AI models
- **Observability**: Prometheus metrics and structured logging

## Prerequisites

- Go 1.21 or higher
- (Optional) NVIDIA GPU with CUDA 12.2+ for GPU challenge support
- (Optional) Docker with NVIDIA Container Toolkit for containerized GPU deployment
- (Optional) [direnv](https://direnv.net/) for managing environment variables
- Access to Ethereum RPC endpoint
- OpenAI-compatible LLM backend(s)

## Quick Start

1.  **Copy Templates:**

    Create your `model_backend.yaml` and `config.yaml` from the provided templates.

    ```bash
    cp model_backend.yaml.template model_backend.yaml
    cp config.yaml.template config.yaml
    ```

2.  **Generate Node Key:**

    Create a new account (keypair) for your node.

    ```bash
    go run github.com/fxnlabs/function-node/cmd/fxn account new
    ```

3.  *Register node into the network using the address**

    Create a new account (keypair) for your node.

    ```bash
    go run github.com/fxnlabs/function-node/cmd/fxn account get
    ```

    This will output your node address.

4.  **Run the Node:**

    Start the node software.

    ```bash
    go run github.com/fxnlabs/function-node/cmd/fxn start
    ```

## Implementation Details

For a detailed explanation of the Function Node's implementation, please see the [implementation details](implementation.md).

To check the current implementation status, refer to the [implementation status](implementation_status.md).

## Testing

To run the test suite, use the following command:

```bash
go test ./...
```

### Mocks

This project uses `mockery` to generate mocks for interfaces. To generate mocks, first install `mockery`:

```bash
go install github.com/vektra/mockery/v2@v2.43.2
```

Then, run the following command to generate the mocks based on the `.mockery.yaml` configuration file:

```bash
mockery
```

## Using the `send_request` Command for Testing
This command is a helper to help SHA256 and send a request to your node for testing purposes.

1.  **Set your private key:**

    Export your hex-encoded private key as an environment variable. For gateway requests, use your gateway key. For challenge requests, use the scheduler key.

    ```bash
    export PRIVATE_KEY=your_private_key_here
    ```

2.  **Run the command:**

    The command takes two arguments: the endpoint and the request body.

    

    **Gateway Request Examples:**

    For gateway requests, you'll typically be calling the OpenAI proxy endpoints.

    The private key for test gateway is available in `scripts/gateway_test_key.json`.

    ```bash
    export PRIVATE_KEY=$(jq -r .private_key scripts/gateway_test_key.json)
    ```

    ```bash
    # Chat Completions
    go run github.com/fxnlabs/function-node/cmd/send_request /v1/chat/completions '{"model": "meta/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Embeddings
    go run github.com/fxnlabs/function-node/cmd/send_request /v1/embeddings '{"model": "meta/llama-4-scout-17b-16e-instruct", "input": "The quick brown fox jumped over the lazy dog"}'

    # Completions
    go run github.com/fxnlabs/function-node/cmd/send_request /v1/completions '{"model": "meta/llama-4-scout-17b-16e-instruct", "prompt": "Once upon a time"}'

    # Models
    go run github.com/fxnlabs/function-node/cmd/send_request /v1/models '{}'
    ```

    **Challenge Request Examples:**

    To test the `/challenge` endpoint, use the `challenge` request type. The private key for the scheduler is available in `scripts/scheduler_test_key.json`.

    ```bash
    export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
    ```

    ```bash
    # Identity Challenge
    go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{"type": "IDENTITY", "payload": {}}'

    # Matrix Multiplication Challenge (requires GPU)
    go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'

    # Endpoint Reachable Challenge
    go run github.com/fxnlabs/function-node/cmd/send_request /challenge '{"type": "ENDPOINT_REACHABLE", "payload": "https://www.google.com"}'
    ```

## GPU Challenges

The Function Node supports GPU-accelerated challenges to verify computational performance. The primary GPU challenge is matrix multiplication, which tests the provider's ability to perform parallel computations efficiently.

### Matrix Multiplication Challenge

This challenge verifies GPU performance by executing large matrix multiplications:

- **Small matrices (< 64x64)**: Executed on CPU for efficiency
- **Large matrices (≥ 64x64)**: Executed on GPU for performance verification
- **Automatic fallback**: Uses CPU if GPU is unavailable

### GPU Setup

For GPU support, you'll need:

1. **NVIDIA GPU**: CUDA Compute Capability 5.0 or higher
2. **CUDA Toolkit**: Version 12.2 or compatible
3. **Environment Setup**: Use direnv to automatically set the LD_LIBRARY_PATH
   ```bash
   # The project includes a .envrc file that automatically configures
   # the CUDA library path when you enter the directory
   direnv allow
   ```
4. **Build with CUDA support**:
   ```bash
   make cuda
   ```

For detailed GPU setup and deployment instructions, see [docs/matrix-multiplication-deployment.md](docs/matrix-multiplication-deployment.md).

### Docker Deployment with GPU

```bash
# Build CUDA-enabled image
docker build -f Dockerfile.cuda -t function-node:cuda .

# Run with GPU support
docker-compose -f docker-compose.cuda.yml up
```
