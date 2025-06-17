# Function Node

### Using the `send_request` Command
This command is a helper to help SHA256 and send a request to your node for testing purposes.

1.  **Set your private key:**

    Export your hex-encoded private key as an environment variable. For gateway requests, use your gateway key. For challenge requests, use the scheduler key.

    ```bash
    export PRIVATE_KEY=your_private_key_here
    ```

2.  **Run the command:**

    The command takes two arguments: the request type (`gateway` or `challenge`) and the request body.

    **Gateway Request Examples:**

    For gateway requests, you'll typically be calling the OpenAI proxy endpoints.

    ```bash
    # Chat Completions
    go run cmd/send_request/main.go gateway '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Embeddings
    go run cmd/send_request/main.go gateway '{"model": "text-embedding-ada-002", "input": "The quick brown fox jumped over the lazy dog"}'
    ```

    **Challenge Request Examples:**

    To test the `/challenge` endpoint, use the `challenge` request type. The private key for the scheduler is available in `scripts/scheduler_test_key.json`.

    ```bash
    export PRIVATE_KEY=$(jq -r .private_key scripts/scheduler_test_key.json)
    ```

    **Identity Challenge:**
    ```bash
    go run cmd/send_request/main.go challenge '{"type": "IDENTITY", "payload": {}}'
    ```

    **Matrix Multiplication Challenge:**
    ```bash
    go run cmd/send_request/main.go challenge '{"type": "MATRIX_MULTIPLICATION", "payload": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}}'
    ```

    **Endpoint Reachable Challenge:**
    ```bash
    go run cmd/send_request/main.go challenge '{"type": "ENDPOINT_REACHABLE", "payload": "https://www.google.com"}'
    ```
