#!/bin/bash
# Detect available GPU backend

# Function to check CUDA availability
check_cuda() {
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        # Check if CUDA is installed
        if [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ] || command -v nvcc &> /dev/null; then
            echo "cuda"
            return 0
        fi
    fi
    return 1
}

# Function to check Metal availability (macOS)
check_metal() {
    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Check if Metal framework is available
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
            echo "metal"
            return 0
        fi
    fi
    return 1
}

# Main detection logic
if [ "$1" != "" ]; then
    # If a specific backend is requested, validate it
    case "$1" in
        cuda)
            if check_cuda; then
                exit 0
            else
                echo "cuda backend requested but not available" >&2
                exit 1
            fi
            ;;
        metal)
            if check_metal; then
                exit 0
            else
                echo "metal backend requested but not available" >&2
                exit 1
            fi
            ;;
        cpu)
            echo "cpu"
            exit 0
            ;;
        auto|*)
            # Auto-detect best available backend
            if check_cuda; then
                exit 0
            elif check_metal; then
                exit 0
            else
                echo "cpu"
                exit 0
            fi
            ;;
    esac
else
    # Default to auto-detection
    if check_cuda; then
        exit 0
    elif check_metal; then
        exit 0
    else
        echo "cpu"
        exit 0
    fi
fi