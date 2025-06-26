# Function Node Makefile
.PHONY: all build cuda metal cpu test benchmark clean docker-cuda docker-cpu

# Default target
all: build

# Build with CUDA support
cuda: cuda-compile
	@echo "Building with CUDA support..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I/usr/local/cuda/include -I./cuda" \
	CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L./cuda -lmatmul_cuda -lcuda -lcudart -lcublas -lstdc++" \
	go build -tags cuda -o fxn github.com/fxnlabs/function-node/cmd/fxn

# Compile CUDA sources
cuda-compile:
	@echo "Compiling CUDA sources..."
	$(MAKE) -C cuda all

# Build with Metal support (macOS only)
metal: metal-compile
	@echo "Building with Metal support..."
	@echo "Metal library compiled, now building Go binary with embedded shader..."
	CGO_ENABLED=1 \
	go build -tags metal -o fxn github.com/fxnlabs/function-node/cmd/fxn

# Compile Metal sources and shaders
metal-compile:
	@echo "Compiling Metal backend..."
	@if [ ! -f metal/lib/matmul.metallib ]; then \
		echo "Metal library not found, compiling..."; \
		$(MAKE) -C metal all; \
	else \
		echo "Metal library already exists. Run 'make clean' to rebuild."; \
	fi

# Build with CPU fallback (no GPU)
cpu:
	@echo "Building with CPU fallback..."
	go build -o fxn github.com/fxnlabs/function-node/cmd/fxn

# Default build (CPU)
build: cpu

# Run tests with CUDA support
test:
	@echo "Running tests..."
	cd cuda && make install-local
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I/usr/local/cuda/include" \
	CGO_LDFLAGS="-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas" \
	LD_LIBRARY_PATH="$(PWD)/cuda/lib:$$LD_LIBRARY_PATH" \
	go test -tags cuda -v $$(go list ./... | grep -v /examples)

# Run tests without CUDA
test-cpu:
	@echo "Running CPU tests..."
	go test -v ./...

# Run tests with Metal support
test-metal:
	@echo "Running Metal tests..."
	CGO_ENABLED=1 \
	DYLD_LIBRARY_PATH="$(PWD)/metal/lib:$$DYLD_LIBRARY_PATH" \
	go test -tags metal -v ./internal/gpu/...

# Run performance benchmarks
benchmark:
	@echo "Running benchmarks..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I/usr/local/cuda/include" \
	CGO_LDFLAGS="-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas" \
	go test -tags cuda -bench=. -benchmem ./internal/challenge/...

# Run CPU benchmarks
benchmark-cpu:
	@echo "Running CPU benchmarks..."
	go test -bench=. -benchmem ./internal/challenge/...

# Run Metal benchmarks
benchmark-metal:
	@echo "Running Metal benchmarks..."
	CGO_ENABLED=1 \
	go test -tags metal -bench=. -benchmem ./internal/gpu/...

# Build Docker image with CUDA support
docker-cuda:
	@echo "Building Docker image with CUDA support..."
	docker build -f Dockerfile.cuda -t function-node:cuda .

# Build Docker image with CPU support
docker-cpu:
	@echo "Building Docker image with CPU support..."
	docker build -f Dockerfile -t function-node:latest .

# Run with docker-compose (CUDA)
docker-run-cuda:
	docker-compose -f docker-compose.cuda.yml up

# Run with docker-compose (CPU)
docker-run-cpu:
	docker-compose up

# Generate mocks
mocks:
	@echo "Generating mocks..."
	mockery

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f fxn
	$(MAKE) -C cuda clean 2>/dev/null || true
	$(MAKE) -C metal clean 2>/dev/null || true
	go clean -cache

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod tidy

# Lint code
lint:
	@echo "Running linter..."
	golangci-lint run

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null 2>&1 && echo "CUDA compiler found at: $$(which nvcc)" || echo "CUDA compiler not found"
	@which nvidia-smi > /dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"

# Check Metal availability (macOS only)
check-metal:
	@echo "Checking Metal availability..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Running on macOS - Metal support available"; \
		system_profiler SPDisplaysDataType | grep -E "(Chipset Model:|VRAM|GPU|Graphics)"; \
		CGO_ENABLED=1 go run -tags metal github.com/fxnlabs/function-node/cmd/fxn gpu info 2>/dev/null || echo "Run 'make metal' to build with Metal support"; \
	else \
		echo "Metal is only available on macOS"; \
	fi

# Development build with hot reload (requires air)
dev:
	@echo "Starting development server with hot reload..."
	air

# Help
help:
	@echo "Available targets:"
	@echo "  cuda          - Build with CUDA support"
	@echo "  metal         - Build with Metal support (macOS only)"
	@echo "  metal-compile - Compile Metal backend library and shaders"
	@echo "  cpu           - Build with CPU fallback"
	@echo "  test          - Run tests with CUDA support"
	@echo "  test-cpu      - Run tests without GPU"
	@echo "  test-metal    - Run tests with Metal support"
	@echo "  benchmark     - Run performance benchmarks with CUDA"
	@echo "  benchmark-cpu - Run performance benchmarks without GPU"
	@echo "  benchmark-metal - Run performance benchmarks with Metal"
	@echo "  docker-cuda   - Build Docker image with CUDA support"
	@echo "  docker-cpu    - Build Docker image with CPU support"
	@echo "  docker-run-cuda - Run with docker-compose (CUDA)"
	@echo "  docker-run-cpu  - Run with docker-compose (CPU)"
	@echo "  mocks         - Generate mocks"
	@echo "  clean         - Clean build artifacts"
	@echo "  deps          - Install dependencies"
	@echo "  lint          - Run linter"
	@echo "  fmt           - Format code"
	@echo "  check-cuda    - Check CUDA installation"
	@echo "  check-metal   - Check Metal availability"
	@echo "  dev           - Start development server with hot reload"
	@echo "  help          - Show this help message"
