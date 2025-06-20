#!/bin/bash

# Script to test the matrix multiplication challenge endpoint
# This sends various test requests to validate the challenge functionality

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load scheduler test key
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIVATE_KEY=$(jq -r .private_key "${SCRIPT_DIR}/scheduler_test_key.json")
ADDRESS=$(jq -r .address "${SCRIPT_DIR}/scheduler_test_key.json")

if [ -z "$PRIVATE_KEY" ] || [ -z "$ADDRESS" ]; then
    echo -e "${RED}Error: Failed to load scheduler test key${NC}"
    exit 1
fi

echo -e "${GREEN}Testing Matrix Multiplication Challenge${NC}"
echo "Scheduler Address: ${ADDRESS}"
echo ""

# Function to send a challenge request
send_challenge() {
    local name=$1
    local payload=$2

    echo -e "${YELLOW}Test: ${name}${NC}"

    # Use the send_request tool
    response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
        "/challenge" \
        "$payload" 2>&1)

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
        comp_time=$(echo "$response" | jq -r '.computationTimeMs // 0')
        flops=$(echo "$response" | jq -r '.flops // 0')
        backend_used=$(echo "$response" | jq -r '.backend // ""')
        if [ "$comp_time" != "0" ] && [ "$flops" != "0" ]; then
            gflops=$(echo "scale=3; $flops / ($comp_time / 1000) / 1000000000" | bc 2>/dev/null || echo "0")
            echo -e "Computation Time: ${comp_time}ms, GFLOPS: ${gflops}, Backend: ${backend_used}"
        else
            echo -e "${RED}Warning: Invalid response format${NC}"
            echo "$response"
        fi
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

# Test 6: Backend selection - CUDA GPU
echo -e "${YELLOW}=== Test 6: CUDA GPU Backend ===${NC}"
send_challenge "Request CUDA GPU backend" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "cuda"
    }
}'

# Test 7: Backend selection - Metal GPU
echo -e "${YELLOW}=== Test 7: Metal GPU Backend ===${NC}"
send_challenge "Request Metal GPU backend" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "metal"
    }
}'

# Test 8: Backend selection - Auto
echo -e "${YELLOW}=== Test 8: Auto Backend ===${NC}"
send_challenge "Auto backend selection" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "size": 50,
        "backend": "auto"
    }
}'

# Test 9: Rectangular matrices
echo -e "${YELLOW}=== Test 9: Rectangular Matrices ===${NC}"
send_challenge "2x3 * 3x2 matrices" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "A": [[1, 2, 3], [4, 5, 6]],
        "B": [[7, 8], [9, 10], [11, 12]]
    }
}'

# Test 10: Error case - incompatible dimensions
echo -e "${YELLOW}=== Test 10: Error - Incompatible Dimensions ===${NC}"
send_challenge "Incompatible dimensions" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {
        "A": [[1, 2]],
        "B": [[3, 4, 5]]
    }
}'

# Test 11: Error case - missing parameters
echo -e "${YELLOW}=== Test 11: Error - Missing Parameters ===${NC}"
send_challenge "No matrices or size" '{
    "type": "MATRIX_MULTIPLICATION",
    "payload": {}
}'

# Backend availability detection
echo -e "${YELLOW}=== Detecting Available Backends ===${NC}"

# Test backend availability with a small matrix
test_response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
    "/challenge" \
    '{"type": "MATRIX_MULTIPLICATION", "payload": {"size": 10, "backend": "cuda"}}' 2>&1)

if [ $? -eq 0 ]; then
    cuda_backend=$(echo "$test_response" | jq -r '.backend // ""')
    if [ "$cuda_backend" == "cuda" ]; then
        CUDA_AVAILABLE=true
        echo -e "CUDA: ${GREEN}Available${NC}"
    else
        CUDA_AVAILABLE=false
        echo -e "CUDA: ${YELLOW}Not available (would fallback to CPU)${NC}"
    fi
else
    CUDA_AVAILABLE=false
    echo -e "CUDA: ${RED}Not available${NC}"
fi

test_response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
    "/challenge" \
    '{"type": "MATRIX_MULTIPLICATION", "payload": {"size": 10, "backend": "metal"}}' 2>&1)

if [ $? -eq 0 ]; then
    metal_backend=$(echo "$test_response" | jq -r '.backend // ""')
    if [ "$metal_backend" == "metal" ]; then
        METAL_AVAILABLE=true
        echo -e "Metal: ${GREEN}Available${NC}"
    else
        METAL_AVAILABLE=false
        echo -e "Metal: ${YELLOW}Not available (would fallback to CPU)${NC}"
    fi
else
    METAL_AVAILABLE=false
    echo -e "Metal: ${RED}Not available${NC}"
fi

echo -e "CPU: ${GREEN}Always available${NC}"
echo ""

# Backend comparison test
echo -e "${YELLOW}=== Backend Comparison Test ===${NC}"
echo "Testing available backends with various matrix sizes..."
echo ""

# Arrays to store results for comparison (using regular arrays with indices)
# We'll use a simple mapping: size_50=0, size_100=1, size_200=2, size_500=3
cpu_times=()
cuda_times=()
metal_times=()
cpu_gflops=()
cuda_gflops=()
metal_gflops=()

# Helper function to get array index for size
get_size_index() {
    case $1 in
    1000) echo 0 ;;
    2000) echo 1 ;;
    4000) echo 2 ;;
    8000) echo 3 ;;
    esac
}

for size in 1000 2000 4000 8000; do
    echo -e "${YELLOW}Testing ${size}x${size} matrices...${NC}"

    # Test CPU backend
    echo -n "  CPU: "
    response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
        "/challenge" \
        "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size, \"backend\": \"cpu\"}}" 2>&1)

    if [ $? -eq 0 ]; then
        comp_time=$(echo "$response" | jq -r '.computationTimeMs // 0')
        flops=$(echo "$response" | jq -r '.flops // 0')
        if [ "$comp_time" != "0" ] && [ "$flops" != "0" ]; then
            gflops=$(echo "scale=3; $flops / ($comp_time / 1000) / 1000000000" | bc 2>/dev/null || echo "0")
            idx=$(get_size_index $size)
            cpu_times[$idx]=$comp_time
            cpu_gflops[$idx]=$gflops
            echo -e "${GREEN}${comp_time}ms (${gflops} GFLOPS)${NC}"
        else
            echo -e "${RED}Failed${NC}"
        fi
    else
        echo -e "${RED}Failed${NC}"
    fi

    # Test CUDA backend (only if available)
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo -n "  CUDA: "
        response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
            "/challenge" \
            "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size, \"backend\": \"cuda\"}}" 2>&1)

        if [ $? -eq 0 ]; then
            comp_time=$(echo "$response" | jq -r '.computationTimeMs // 0')
            flops=$(echo "$response" | jq -r '.flops // 0')
            backend_used=$(echo "$response" | jq -r '.backend // ""')

            if [ "$backend_used" == "cpu" ]; then
                echo -e "${YELLOW}Fallback to CPU${NC}"
            elif [ "$comp_time" != "0" ] && [ "$flops" != "0" ]; then
                gflops=$(echo "scale=3; $flops / ($comp_time / 1000) / 1000000000" | bc 2>/dev/null || echo "0")
                idx=$(get_size_index $size)
                cuda_times[$idx]=$comp_time
                cuda_gflops[$idx]=$gflops
                speedup=$(echo "scale=2; ${cpu_times[$idx]} / $comp_time" | bc 2>/dev/null || echo "N/A")
                echo -e "${GREEN}${comp_time}ms (${gflops} GFLOPS, ${speedup}x speedup)${NC}"
            else
                echo -e "${RED}Failed${NC}"
            fi
        else
            echo -e "${RED}Failed${NC}"
        fi
    else
        echo -e "  CUDA: ${YELLOW}Skipped (not available)${NC}"
    fi

    # Test Metal backend (only if available)
    if [ "$METAL_AVAILABLE" = true ]; then
        echo -n "  Metal: "
        response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
            "/challenge" \
            "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size, \"backend\": \"metal\"}}" 2>&1)

        if [ $? -eq 0 ]; then
            comp_time=$(echo "$response" | jq -r '.computationTimeMs // 0')
            flops=$(echo "$response" | jq -r '.flops // 0')
            backend_used=$(echo "$response" | jq -r '.backend // ""')

            if [ "$backend_used" == "cpu" ]; then
                echo -e "${YELLOW}Fallback to CPU${NC}"
            elif [ "$comp_time" != "0" ] && [ "$flops" != "0" ]; then
                gflops=$(echo "scale=3; $flops / ($comp_time / 1000) / 1000000000" | bc 2>/dev/null || echo "0")
                idx=$(get_size_index $size)
                metal_times[$idx]=$comp_time
                metal_gflops[$idx]=$gflops
                speedup=$(echo "scale=2; ${cpu_times[$idx]} / $comp_time" | bc 2>/dev/null || echo "N/A")
                echo -e "${GREEN}${comp_time}ms (${gflops} GFLOPS, ${speedup}x speedup)${NC}"
            else
                echo -e "${RED}Failed${NC}"
            fi
        else
            echo -e "${RED}Failed${NC}"
        fi
    else
        echo -e "  Metal: ${YELLOW}Skipped (not available)${NC}"
    fi

    echo ""
done

# Print summary comparison table
echo -e "${YELLOW}=== Performance Summary ===${NC}"

# Build dynamic table header based on available backends
table_header="| Size   | CPU (ms)    "
table_divider="+--------+-------------"
if [ "$CUDA_AVAILABLE" = true ]; then
    table_header="${table_header}| CUDA (ms)   "
    table_divider="${table_divider}+-------------"
fi
if [ "$METAL_AVAILABLE" = true ]; then
    table_header="${table_header}| Metal (ms)  "
    table_divider="${table_divider}+-------------"
fi
table_header="${table_header}|"
table_divider="${table_divider}+"

echo "$table_divider"
echo "$table_header"
echo "$table_divider"

for size in 1000 2000 4000 8000; do
    idx=$(get_size_index $size)
    cpu_str="${cpu_times[$idx]:-N/A}"

    # Build row dynamically
    row_data="| %-6s | %-11s "
    row_values=("$size" "$cpu_str")

    if [ "$CUDA_AVAILABLE" = true ]; then
        cuda_str="${cuda_times[$idx]:-N/A}"
        # Add speedup info if available
        if [ "${cuda_times[$idx]}" != "" ] && [ "${cpu_times[$idx]}" != "" ]; then
            speedup=$(echo "scale=1; ${cpu_times[$idx]} / ${cuda_times[$idx]}" | bc 2>/dev/null)
            cuda_str="${cuda_str} (${speedup}x)"
        fi
        row_data="${row_data}| %-11s "
        row_values+=("$cuda_str")
    fi

    if [ "$METAL_AVAILABLE" = true ]; then
        metal_str="${metal_times[$idx]:-N/A}"
        # Add speedup info if available
        if [ "${metal_times[$idx]}" != "" ] && [ "${cpu_times[$idx]}" != "" ]; then
            speedup=$(echo "scale=1; ${cpu_times[$idx]} / ${metal_times[$idx]}" | bc 2>/dev/null)
            metal_str="${metal_str} (${speedup}x)"
        fi
        row_data="${row_data}| %-11s "
        row_values+=("$metal_str")
    fi

    row_data="${row_data}|"
    # shellcheck disable=SC2059
    printf "$row_data\n" "${row_values[@]}"
done
echo "$table_divider"

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

            if response=$(PRIVATE_KEY="$PRIVATE_KEY" go run "${SCRIPT_DIR}/../cmd/send_request/main.go" \
                "/challenge" \
                "{\"type\": \"MATRIX_MULTIPLICATION\", \"payload\": {\"size\": $size}}" 2>&1); then
                comp_time=$(echo "$response" | jq -r '.computationTimeMs // 0')
                backend=$(echo "$response" | jq -r '.backend // "unknown"')
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
