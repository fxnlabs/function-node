#!/bin/bash

# Script to test the matrix multiplication challenge endpoint
# This sends various test requests to validate the challenge functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
HOST="${HOST:-localhost}"
PORT="${PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"

# Load scheduler test key
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIVATE_KEY=$(jq -r .private_key "${SCRIPT_DIR}/scheduler_test_key.json")
ADDRESS=$(jq -r .address "${SCRIPT_DIR}/scheduler_test_key.json")

if [ -z "$PRIVATE_KEY" ] || [ -z "$ADDRESS" ]; then
    echo -e "${RED}Error: Failed to load scheduler test key${NC}"
    exit 1
fi

echo -e "${GREEN}Testing Matrix Multiplication Challenge${NC}"
echo "Server: ${BASE_URL}"
echo "Scheduler Address: ${ADDRESS}"
echo ""

# Function to send a challenge request
send_challenge() {
    local name=$1
    local payload=$2
    
    echo -e "${YELLOW}Test: ${name}${NC}"
    
    # Use the send_request tool
    response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
        -url "${BASE_URL}/challenge" \
        -data "$payload" 2>&1)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
        echo "$response" | jq '.' || echo "$response"
    else
        echo -e "${RED}✗ Failed${NC}"
        echo "$response"
    fi
    echo ""
}

# Test 1: Small fixed matrices
echo -e "${YELLOW}=== Test 1: Small Fixed Matrices ===${NC}"
send_challenge "2x2 matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "A": [[1, 2], [3, 4]],
        "B": [[5, 6], [7, 8]]
    }
}'

# Test 2: Random matrix generation (small)
echo -e "${YELLOW}=== Test 2: Random Matrix Generation (10x10) ===${NC}"
send_challenge "10x10 random matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 10
    }
}'

# Test 3: Medium size matrices
echo -e "${YELLOW}=== Test 3: Medium Size Matrices (100x100) ===${NC}"
send_challenge "100x100 matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 100
    }
}'

# Test 4: Large matrices (result not included)
echo -e "${YELLOW}=== Test 4: Large Matrices (200x200) ===${NC}"
send_challenge "200x200 matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 200
    }
}'

# Test 5: Backend selection - CPU
echo -e "${YELLOW}=== Test 5: CPU Backend ===${NC}"
send_challenge "Force CPU backend" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "cpu"
    }
}'

# Test 6: Backend selection - GPU
echo -e "${YELLOW}=== Test 6: GPU Backend ===${NC}"
send_challenge "Request GPU backend" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "gpu"
    }
}'

# Test 7: Backend selection - Auto
echo -e "${YELLOW}=== Test 7: Auto Backend ===${NC}"
send_challenge "Auto backend selection" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "auto"
    }
}'

# Test 8: Rectangular matrices
echo -e "${YELLOW}=== Test 8: Rectangular Matrices ===${NC}"
send_challenge "2x3 * 3x2 matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "A": [[1, 2, 3], [4, 5, 6]],
        "B": [[7, 8], [9, 10], [11, 12]]
    }
}'

# Test 9: Error case - incompatible dimensions
echo -e "${YELLOW}=== Test 9: Error - Incompatible Dimensions ===${NC}"
send_challenge "Incompatible dimensions" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "A": [[1, 2]],
        "B": [[3, 4, 5]]
    }
}'

# Test 10: Error case - missing parameters
echo -e "${YELLOW}=== Test 10: Error - Missing Parameters ===${NC}"
send_challenge "No matrices or size" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {}
}'

# Performance test with different sizes
echo -e "${YELLOW}=== Performance Test ===${NC}"
for size in 50 100 200 500; do
    echo -e "${YELLOW}Testing ${size}x${size} matrices...${NC}"
    
    start_time=$(date +%s.%N)
    
    response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
        -url "${BASE_URL}/challenge" \
        -data "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size}}" 2>&1)
    
    end_time=$(date +%s.%N)
    total_time=$(echo "$end_time - $start_time" | bc)
    
    if [ $? -eq 0 ]; then
        # Extract computation time and calculate GFLOPS
        comp_time=$(echo "$response" | jq -r '.result.computationTimeMs // 0')
        flops=$(echo "$response" | jq -r '.result.flops // 0')
        backend=$(echo "$response" | jq -r '.result.backend // "unknown"')
        
        if [ "$comp_time" != "0" ] && [ "$flops" != "0" ]; then
            gflops=$(echo "scale=2; $flops / ($comp_time / 1000) / 1000000000" | bc)
            echo -e "${GREEN}✓ Size: ${size}x${size}, Backend: ${backend}, Computation: ${comp_time}ms, GFLOPS: ${gflops}, Total: ${total_time}s${NC}"
        else
            echo -e "${GREEN}✓ Size: ${size}x${size}, Total: ${total_time}s${NC}"
        fi
    else
        echo -e "${RED}✗ Failed for size ${size}${NC}"
    fi
done

echo ""
echo -e "${GREEN}Matrix multiplication challenge tests completed!${NC}"

# Optional: Run continuous performance monitoring
if [ "$1" == "--monitor" ]; then
    echo ""
    echo -e "${YELLOW}=== Continuous Performance Monitoring ===${NC}"
    echo "Press Ctrl+C to stop"
    
    while true; do
        for size in 100 200 500; do
            echo -n "Testing ${size}x${size}... "
            
            start_time=$(date +%s.%N)
            response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
                -url "${BASE_URL}/challenge" \
                -data "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size}}" 2>&1)
            
            if [ $? -eq 0 ]; then
                comp_time=$(echo "$response" | jq -r '.result.computationTimeMs // 0')
                backend=$(echo "$response" | jq -r '.result.backend // "unknown"')
                echo "Backend: ${backend}, Time: ${comp_time}ms"
            else
                echo "Failed"
            fi
            
            sleep 1
        done
        echo "---"
        sleep 5
    done
fi