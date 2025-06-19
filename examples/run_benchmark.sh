#!/bin/bash

# Function Node Matrix Multiplication Benchmark Runner
# This script runs the matrix multiplication challenge demo with proper CUDA setup

set -e

echo "🔧 Setting up CUDA environment..."

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Build CUDA library if needed
if [ ! -f "cuda/lib/libmatmul_cuda.so" ]; then
    echo "📦 Building CUDA library..."
    cd cuda && make install-local && cd ..
fi

echo "🚀 Running Matrix Multiplication Benchmark..."

# Run the benchmark with proper environment
CGO_ENABLED=1 \
CGO_CFLAGS="-I/usr/local/cuda/include" \
CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L./cuda/lib -lcuda -lcudart -lcublas -lmatmul_cuda" \
LD_LIBRARY_PATH="./cuda/lib:$LD_LIBRARY_PATH" \
go run -tags cuda ./examples/test_matrix_challenge.go

echo ""
echo "✨ Benchmark completed! The results show CPU vs GPU performance comparison."
echo "💡 Higher GFLOPS = better performance, higher speedup = better GPU acceleration"