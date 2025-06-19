#!/bin/bash

# Function Node Matrix Multiplication Benchmark Runner
# This script runs the matrix multiplication challenge demo with GPU support (CUDA or Metal)

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Detect platform and available GPU
OS=$(uname)
GPU_TYPE=""

# Check for NVIDIA GPU (CUDA)
if command -v nvidia-smi &> /dev/null; then
    GPU_TYPE="cuda"
elif [ "$OS" = "Darwin" ]; then
    # Check for Apple Silicon
    if [ "$(uname -m)" = "arm64" ] || sysctl -n machdep.cpu.brand_string | grep -q "Apple"; then
        GPU_TYPE="metal"
    fi
fi

if [ -z "$GPU_TYPE" ]; then
    echo "⚠️  No supported GPU detected. Running CPU-only benchmark..."
    go run ./examples/test_matrix_challenge.go
    exit 0
fi

echo "🔧 Detected GPU type: $(echo $GPU_TYPE | tr '[:lower:]' '[:upper:]')"

if [ "$GPU_TYPE" = "cuda" ]; then
    echo "🔧 Setting up CUDA environment..."
    
    # Build CUDA library if needed
    if [ ! -f "cuda/lib/libmatmul_cuda.so" ]; then
        echo "📦 Building CUDA library..."
        cd cuda && make install-local && cd ..
    fi
    
    echo "🚀 Running Matrix Multiplication Benchmark (CUDA vs CPU)..."
    
    # Run the benchmark with CUDA environment
    CGO_ENABLED=1 \
    CGO_CFLAGS="-I/usr/local/cuda/include" \
    CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L./cuda/lib -lcuda -lcudart -lcublas -lmatmul_cuda" \
    LD_LIBRARY_PATH="./cuda/lib:$LD_LIBRARY_PATH" \
    go run -tags cuda ./examples/test_matrix_challenge.go
    
elif [ "$GPU_TYPE" = "metal" ]; then
    echo "🔧 Setting up Metal environment..."
    
    # Build Metal backend if needed
    if [ ! -f "metal/lib/libmetal_backend.a" ] || [ ! -f "metal/lib/matmul.metallib" ]; then
        echo "📦 Building Metal backend..."
        make metal-compile
    fi
    
    echo "🚀 Running Matrix Multiplication Benchmark (Metal vs CPU)..."
    
    # Run the benchmark with Metal environment
    CGO_ENABLED=1 \
    go run -tags metal ./examples/test_matrix_challenge.go
fi

echo ""
echo "✨ Benchmark completed! The results show CPU vs GPU performance comparison."
echo "💡 Higher GFLOPS = better performance, higher speedup = better GPU acceleration"