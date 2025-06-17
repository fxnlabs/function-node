# Function Node Software Implementation Status

## Completed

### Config (internal/config)
- [x] Implement a general config yaml file that is read by our code base. This config file will let us configure where the node keys are stored, verbosity.
- [x] Implement a model_backend config yaml which allows node operators to provide model names and endpoints where the model will be proxied to (i.e another http endpoint) when the OpenAI endpoints are called.

### Authentication (internal/auth)
- [x] Implement a provider registry check to ensure the requesting provider is registered with Function Network.
- [x] Implement authentication for challenges endpoint to prevent spoofing.

### OpenAI Endpoints (internal/openai)
- [x] Define a YAML configuration file (`model_backend.yaml`) that allows node operators to configure specific models and point them to different URLs (model backends).
- [x] Implement proxy for chat completion, completion, and embedding endpoints.
- [x] Expose OpenAI endpoints through the node server.

### Challenges Endpoint (internal/challenge)
- [x] Implement periodic challenge endpoint calls from the Scheduler to verify latency, hardware, and overall health.
- [x] Define challenges:
	- [x] **Identity**: Verifies the node's identity and retrieves GPU statistics.
	- [x] **Matrix Multiplication**: Performs a matrix multiplication challenge.
	- [x] **Endpoint Reachable**: Checks if an endpoint is reachable. This will be improved to be a "proxy challenge".

### Smart Contracts (internal/contracts)
- [x] Implement RPC calls to smart contracts using VIEM to grab related registries.
- [x] Cache responses in memory and poll async for updated regsitries periodically.
- [x] Use the provided ABI (Application Binary Interface) for smart contract interactions.

### Observability and Logging (internal/metrics, internal/logger)
- [x] Add promethesus observability features to ensure node operators can monitor their operation.
- [x] Implement logging using the Zap logger, with verbosity configurable via `config.yaml` file.
- [x] Use a consistent logging format throughout the implementation.

### CLI Implementation
- [x] Create a CLI (`fxn accounts new`) that allows users to generate a key file for on-chain registration.
- [x] Handle errors and edge cases when generating the key file.

## In Progress / To Do

### Authentication (internal/auth)
- [x] Implement authentication for OpenAI endpoints to prevent unauthorized access and replay attacks.
- [x] Implement actual challenge authentication in `AuthenticateChallenge`
- [x] Renamed `IsNodeRegistered` to `IsProviderRegistered` (conceptually, to be implemented if needed, using ProviderRegistry).


### Challenges Endpoint (internal/challenge)
- [x] Implement challenge handling logic in `ChallengeHandler` using a strategy pattern.
- [x] Implement GPU stat polling in `internal/challenge/challengers/gpu_stats.go`.
- [x] Implement matrix multiplication challenge in `internal/challenge/challengers/matrix_multiplication.go`.
- [x] Implement endpoint reachability check in `internal/challenge/challengers/endpoint_reachable.go`.
- [x] Separated challenges into their own files under `internal/challenge/challengers` for scalability.

### Smart Contracts (internal/contracts)
- [x] Create the concept of a `CachedRegistry` that calls an RPC provider/smart contracts for updating its caches. This should be a generic implementation to support different registries, with each registry having a configurable update interval.
	- [x] Gateway Registry
	- [x] Scheduler Registry
	- [x] Provider Registry (formerly Node Registry)
- [x] Implement registry fetching for Gateways (using `getActiveGatewaysLive`).
- [x] Make the RPC provider configurable in `config.yaml` and injectable into registries.
- [x] Centralized `ethclient.Client` creation and injection into registry constructors.
- [x] Removed global variables for registries; instances are created in `main` and passed as dependencies.
- [x] Implement actual smart contract fetching logic for ProviderRegistry in `internal/registry/provider_registry.go`.
- [x] Define and use actual ABI for ProviderRegistry.
- [ ] Define and use actual ABI for SchedulerRegistry if contract calls beyond simple address storage are needed.
- [ ] Define and use actual ABI for SchedulerRegistry if contract calls beyond simple address storage are needed.

## Example API Calls

To simplify testing the API, you can use the `send_gateway_request.sh` script. This script automatically generates the required signature and sends the request.

**Note:** The OpenAI endpoints can only be called by registered gateways, and the `/challenge` endpoint can only be called by the registered scheduler.

### Authentication Details

All API requests must include the following headers for authentication:

- `X-Public-Key`: Your hex-encoded public key.
- `X-Timestamp`: A Unix timestamp of when the request was made.
- `X-Nonce`: A unique, randomly generated string for each request to prevent replay attacks.
- `X-Signature`: A signature of the request payload.

The signature is created by signing the following string with your private key:

```
sha256(request_body) + "." + timestamp + "." + nonce
```

### Using the `send_gateway_request.sh` Script
This script is a helper to help SHA256 and send a request to your node for testing purposes.

1.  **Set your private key:**

    Export your hex-encoded private key as an environment variable.

    ```bash
    export PRIVATE_KEY=your_private_key_here
    ```

2.  **Make the script executable:**

    ```bash
    chmod +x ./scripts/send_gateway_request.sh
    ```

3.  **Run the script:**

    The script takes two arguments: the endpoint and the request body.

    **Chat Completions Example:**

    ```bash
    ./scripts/send_gateway_request.sh "/v1/chat/completions" '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
    ```

    **Chat Completions Example (Streaming):**

    ```bash
    ./scripts/send_gateway_request.sh "/v1/chat/completions" '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
    ```

    **Completions Example:**

    ```bash
    ./scripts/send_gateway_request.sh "/v1/completions" '{"model": "text-davinci-003", "prompt": "Once upon a time"}'
    ```

    **Completions Example (Streaming):**

    ```bash
    ./scripts/send_gateway_request.sh "/v1/completions" '{"model": "text-davinci-003", "prompt": "Once upon a time", "stream": true}'
    ```

    **Embeddings Example:**

    ```bash
    ./scripts/send_gateway_request.sh "/v1/embeddings" '{"model": "text-embedding-ada-002", "input": "The quick brown fox jumped over the lazy dog"}'
    ```

    ### Challenge Examples

    To test the `/challenge` endpoint, you can use the `send_challenge_request.sh` script. This script handles the authentication and signing of the request.

    **Note:** The `/challenge` endpoint can only be called by the registered scheduler. The private key for the scheduler is available in `scripts/scheduler_test_key.json`.

    1.  **Set your private key:**

        Export the scheduler's private key as an environment variable.

        ```bash
        export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
        ```

    2.  **Make the script executable:**

        ```bash
        chmod +x ./scripts/send_challenge_request.sh
        ```

    3.  **Run the script with the desired challenge:**

        The script takes one argument: the request body. The `type` field in the JSON payload determines which challenge to run.

        **Identity Challenge:**

        This challenge verifies the node's identity. The payload is not used.

        ```bash
        ./scripts/send_challenge_request.sh '{"type": "IDENTITY", "payload": {}}'
        ```

        **Matrix Multiplication Challenge:**

        This challenge performs a matrix multiplication. The payload must contain two matrices, `A` and `B`.

        ```bash
        ./scripts/send_challenge_request.sh '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'
        ```

        **Endpoint Reachable Challenge:**

        This challenge checks if an endpoint is reachable. The payload must be the URL of the endpoint to check.

        ```bash
        ./scripts/send_challenge_request.sh '{"type": "ENDPOINT_REACHABLE", "payload": "https://www.google.com"}'
        ```
