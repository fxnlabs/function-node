# Function Node Software 

## Summary

The Function Node software is a P2P (Peer-to-Peer) system built for node 
operators of Function Network. It acts as a proxy to LLM (Large Language 
Model) inference endpoints and supports OpenAI-compatible endpoints.

### Key Components
1. **Authentication**: Verify node registration, prevent spoofing, and 
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
1. Ensure node operators can monitor their operation and receive alerts 
when issues arise.
2. Prevent spoofing and ensure authenticity of requests from the Scheduler 
entity.
3. Improve performance by caching responses in memory and using RPC calls 
for smart contract interactions.


## Config (internal/config)
* Implement a general config yaml file that is read by our code base. This config file will let us configure where the node keys are stored, verbosity.

* Implement a backend config yaml which allows node operators to provide model names and endpoints where the model will be proxied to (i.e another http endpoint) when the OpenAI endpoints are called. More information can be found in the OpenAI endpoint section

## Authentication (internal/auth)

* Implement a node registry check to ensure the requesting node is
registered with Function Network.
    + If a node is not registered, keep polling until it is registered.
* Implement authentication for challenges endpoint to prevent spoofing by verifying a signature against a message and public key.
* Implement authentication for OpenAI endpoints:
    + Verify requests come from an authorized gateway by checking for a valid signature in the `X-Signature` header.
    + The signature is created from the SHA-256 hash of the request body, a timestamp from the `X-Timestamp` header, and a nonce from the `X-Nonce` header.
    + This approach protects against unauthorized access and replay attacks.
    + A nonce cache is used to prevent replay attacks within the valid timestamp window.

## OpenAI Endpoints (internal/oepnai)

* Define a YAML configuration file (`backend.yaml`) that allows node 
operators to configure specific models and point them to different URLs 
(backends).
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
	1. **Poll GPU Stats**: Polls metadata of GPUs provided by node operators.
	2. **Matrix Multiplication**: Performs a matrix multiplication challenge to ensure the authenticity of node operator GPU responses and speed.
	3. **Poll Endpoint Reachable**: Basic poll checks to ensure endpoints are reachable.

## Smart Contracts (internal/contracts)

* Implement RPC calls to smart contracts using VIEM to grab related registries.
* Cache responses in memory and poll async for updated regsitries periodically. The registry poll checks can be configured via our general config library. Use a RPC library to cache if VIEM doesnt support by default.
* Use the provided ABI (Application Binary Interface) for smart contract interactions.

## Observability and Logging (internal/metrics, internal/logger)

* Add promethesus observability features to ensure node operators can monitor their operation:
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
