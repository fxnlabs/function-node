# Function Node Software 

## Summary

The Function Node software is a P2P (Peer-to-Peer) system built for provider 
operators (formerly node operators) of Function Network. It acts as a proxy to LLM (Large Language 
Model) inference endpoints and supports OpenAI-compatible endpoints.

### Key Components
1. **Authentication**: Verify provider registration, prevent spoofing, and 
authenticate requests from the Scheduler entity.
2. **OpenAI Endpoints**: Configure specific models and point them to 
different URLs via a YAML configuration file.
3. **Challenges Endpoint**: Periodically call challenges from the 
Scheduler to verify latency, hardware, and overall health.
4. **Smart Contracts**: Use VIEM to interact with smart contracts, cache 
responses in memory, and use RPC calls for better performance improvement.
5. **Observability and Logging**: Add observability features and implement 
logging using Zap logger.

**Key Goals:**
1. Ensure provider operators can monitor their operation and receive alerts 
when issues arise.
2. Prevent spoofing and ensure authenticity of requests from the Scheduler 
entity.
3. Improve performance by caching responses in memory and using RPC calls 
for smart contract interactions.


## Config (internal/config)
* Implement a general config yaml file that is read by our code base. This config file will let us configure where the provider keys are stored, verbosity, and RPC provider details. It also includes configuration for polling intervals and smart contract addresses for various registries (Gateway, Scheduler, Provider).

* Implement a model_backend config yaml which allows provider operators to provide model names and endpoints where the model will be proxied to (i.e another http endpoint) when the OpenAI endpoints are called. More information can be found in the OpenAI endpoint section

## Authentication (internal/auth)

* Implement a provider registry check (using `ProviderRegistry`) to ensure the requesting provider is
registered with Function Network.
    + If a provider is not registered, the system should handle this appropriately (e.g., deny access or trigger alerts, actual polling behavior TBD based on `ProviderRegistry` implementation).
* Implement authentication for challenges endpoint to prevent spoofing by verifying a signature against a message and public key.
* Implement authentication for OpenAI endpoints:
    + Verify requests come from an authorized gateway by checking for a valid signature in the `X-Signature` header.
    + The signature is created from the SHA-256 hash of the request body, a timestamp from the `X-Timestamp` header, and a nonce from the `X-Nonce` header.
    + This approach protects against unauthorized access and replay attacks.
    + A nonce cache is used to prevent replay attacks within the valid timestamp window.

## OpenAI Endpoints (internal/openai)

* Define a YAML configuration file (`model_backend.yaml`) that allows provider 
operators to configure specific models and point them to different URLs 
(model backends).
* Include the following endpoints:
	1. Chat completion (streaming and non-streaming)
	2. Completion
	3. Embedding

## Challenges Endpoint (internal/challenge)

* Implement periodic challenge endpoint calls from the Scheduler to verify latency, hardware, and overall health.
* The challenge system is designed using a strategy pattern to allow for easy extensibility.
* A central `ChallengeHandler` receives all challenge requests and delegates them to the appropriate challenger based on the challenge type.
* Each challenge is implemented in its own file in the `internal/challenge/challengers` directory.
* Defined challenges:
	1. **Poll GPU Stats**: Polls metadata of GPUs provided by provider operators.
	2. **Matrix Multiplication**: Performs a matrix multiplication challenge to ensure the authenticity of provider operator GPU responses and speed.
	3. **Poll Endpoint Reachable**: Basic poll checks to ensure endpoints are reachable.

## Smart Contracts (internal/registry, internal/contracts)

* Implement RPC calls to smart contracts using `go-ethereum/ethclient` to grab related registries (Gateway, Scheduler, Provider).
* A generic `CachedRegistry` (`internal/registry/registry.go`) handles caching responses in memory and polling asynchronously for updated registries.
    + Each specific registry (e.g., `GatewayRegistry`, `ProviderRegistry`) implements a `FetchFunc` to retrieve its data.
    + Polling intervals and smart contract addresses are configurable via `config.yaml`.
    + An `ethclient.Client` is instantiated once in `main` and injected into registry constructors.
* Use provided ABIs (Application Binary Interface) for smart contract interactions (e.g., `GatewayRegistry.json`).
* `GatewayRegistry` fetches active gateways using `getActiveGatewaysLive`.
* `ProviderRegistry` and `SchedulerRegistry` currently have dummy fetch implementations; these need to be updated to interact with their respective smart contracts.
* The scheduler registry is currently not implemented on purpose. Because right now it is actually an EOA that is hardcoded, can you add this into our README / implementation.md that this is on purpose as there will only be one scheduler right now

## Observability and Logging (internal/metrics, internal/logger)

* Add promethesus observability features to ensure provider operators can monitor their operation:
	+ Scheduler checks
	+ Endpoint responses and error codes
* Implement logging using the Zap logger, with verbosity configurable via 
`config.yaml` file.
* Use a consistent logging format throughout the implementation.

## CLI Implementation

* Create a CLI (`fxn accounts new`) that allows users to generate a key 
file for on-chain registration.
* Handle errors and edge cases when generating the key file.

Some additional considerations:
* Consider using established libraries or frameworks for specific 
components (e.g., authentication, logging).
* Follow best practices for the Golang code organization, naming conventions, and documentation
