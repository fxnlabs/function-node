# Function Node

The Function Node is the core software for providers on the Function Network, a peer-to-peer system for decentralized AI inference. It serves as a secure, observable proxy to any OpenAI-compatible LLM endpoint, authenticating requests from network gateways and responding to health and performance challenges from schedulers. The node interacts with on-chain registries to stay synced with the network.

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
    go run cmd/cli/main.go account new
    ```

3.  *Register node into the network using the address**

    Create a new account (keypair) for your node.

    ```bash
    go run cmd/cli/main.go account get
    ```

    This will output your node address.

4.  **Run the Node:**

    Start the node software.

    ```bash
    go run cmd/node/main.go
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
    go run cmd/send_request/main.go /v1/chat/completions '{"model": "meta/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Embeddings
    go run cmd/send_request/main.go /v1/embeddings '{"model": "meta/llama-4-scout-17b-16e-instruct", "input": "The quick brown fox jumped over the lazy dog"}'

    # Completions
    go run cmd/send_request/main.go /v1/completions '{"model": "meta/llama-4-scout-17b-16e-instruct", "prompt": "Once upon a time"}'

    # Models
    go run cmd/send_request/main.go /v1/models '{}'
    ```

    **Challenge Request Examples:**

    To test the `/challenge` endpoint, use the `challenge` request type. The private key for the scheduler is available in `scripts/scheduler_test_key.json`.

    ```bash
    export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
    ```

    **Identity Challenge:**
    ```bash
    go run cmd/send_request/main.go /challenge '{"type": "IDENTITY", "payload": {}}'
    ```

    **Matrix Multiplication Challenge:**
    ```bash
    go run cmd/send_request/main.go /challenge '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'
    ```

    **Endpoint Reachable Challenge:**
    ```bash
    go run cmd/send_request/main.go /challenge '{"type": "ENDPOINT_REACHABLE", "payload": "https://www.google.com"}'
    ```
