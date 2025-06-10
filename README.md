# Function Node Software Implementation Status

## Completed

### Config (internal/config)
- [x] Implement a general config yaml file that is read by our code base. This config file will let us configure where the node keys are stored, verbosity.
- [x] Implement a backend config yaml which allows node operators to provide model names and endpoints where the model will be proxied to (i.e another http endpoint) when the OpenAI endpoints are called.

### Authentication (internal/auth)
- [x] Implement a provider registry check to ensure the requesting provider is registered with Function Network.
- [x] Implement authentication for challenges endpoint to prevent spoofing.

### OpenAI Endpoints (internal/openai)
- [x] Define a YAML configuration file (`backend.yaml`) that allows node operators to configure specific models and point them to different URLs (backends).
- [x] Implement proxy for chat completion, completion, and embedding endpoints.
- [x] Expose OpenAI endpoints through the node server.

### Challenges Endpoint (internal/challenge)
- [x] Implement periodic challenge endpoint calls from the Scheduler to verify latency, hardware, and overall health.
- [x] Define challenges:
	- [x] Poll metadata of GPUs provided by node operators
	- [x] Matrix multiplication challenge to ensure authenticity of node operator GPU responses and speed
	- [x] Basic poll checks to ensure endpoints are reachable

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
- [ ] Consider replacing the in-memory nonce cache with a distributed cache (e.g., Redis) for a multi-node setup.
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
- [ ] Implement actual smart contract fetching logic for ProviderRegistry in `internal/registry/provider_registry.go` (currently dummy).
- [ ] Implement actual smart contract fetching logic for SchedulerRegistry in `internal/registry/scheduler_registry.go` (currently returns configured address, may need to fetch dynamic list).
- [ ] Define and use actual ABI for ProviderRegistry if contract calls are needed.
- [ ] Define and use actual ABI for SchedulerRegistry if contract calls beyond simple address storage are needed.
- [ ] Review if `AuthMiddleware` needs to use `ProviderRegistry` for any checks (e.g., `IsProviderRegistered`).
