package challenge

import (
	"fmt"
	"log/slog"
	"math"
	"testing"
	"time"

	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"github.com/fxnlabs/function-node/internal/gpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestMatrixMultiplicationChallenge_Execute(t *testing.T) {
	log := zap.NewNop()
	challenger := challengers.NewMatrixMultiplicationChallenger()

	t.Run("valid multiplication with provided matrices", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
			"B": [][]float64{{5, 6}, {7, 8}},
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)

		// Check response structure
		response, ok := result.(map[string]interface{})
		require.True(t, ok)

		// Verify expected fields exist
		assert.Contains(t, response, "C")
		assert.Contains(t, response, "computationTimeMs")
		assert.Contains(t, response, "backend")
		assert.Contains(t, response, "device")
		assert.Contains(t, response, "matrixSize")
		assert.Contains(t, response, "flops")

		// Verify result matrix
		resultMatrix, ok := response["C"].([][]float64)
		require.True(t, ok)
		assert.Equal(t, [][]float64{{19, 22}, {43, 50}}, resultMatrix)

		// Verify metrics
		assert.Equal(t, 2, response["matrixSize"])
		assert.Equal(t, int64(16), response["flops"]) // 2 * 2^3
	})

	t.Run("random matrix generation", func(t *testing.T) {
		payload := map[string]interface{}{
			"size": 10,
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)

		response, ok := result.(map[string]interface{})
		require.True(t, ok)

		// Verify response structure
		assert.Contains(t, response, "C")
		assert.Contains(t, response, "computationTimeMs")
		assert.Equal(t, 10, response["matrixSize"])
		assert.Equal(t, int64(2000), response["flops"]) // 2 * 10^3

		// Verify result matrix dimensions
		resultMatrix, ok := response["C"].([][]float64)
		require.True(t, ok)
		assert.Equal(t, 10, len(resultMatrix))
		assert.Equal(t, 10, len(resultMatrix[0]))
	})

	t.Run("large matrix without result", func(t *testing.T) {
		payload := map[string]interface{}{
			"size": 200,
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)

		response, ok := result.(map[string]interface{})
		require.True(t, ok)

		// Result should not be included for large matrices
		assert.NotContains(t, response, "C")
		assert.Contains(t, response, "computationTimeMs")
		assert.Equal(t, 200, response["matrixSize"])
	})

	t.Run("backend selection", func(t *testing.T) {
		testCases := []struct {
			name    string
			backend string
		}{
			{"auto backend", "auto"},
			{"cpu backend", "cpu"},
			{"gpu backend", "gpu"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				payload := map[string]interface{}{
					"size":    5,
					"backend": tc.backend,
				}

				result, err := challenger.Execute(payload, log)
				assert.NoError(t, err)

				response, ok := result.(map[string]interface{})
				require.True(t, ok)

				// Backend should be either cpu or gpu
				backend, ok := response["backend"].(string)
				require.True(t, ok)
				assert.Contains(t, []string{"cpu", "gpu", "gpusim", "cuda"}, backend)
			})
		}
	})

	t.Run("incompatible dimensions", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}},
			"B": [][]float64{{3, 4, 5}},
		}

		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("empty matrices", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{},
			"B": [][]float64{},
		}

		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("no size or matrices provided", func(t *testing.T) {
		payload := map[string]interface{}{}

		_, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "either 'size' or both 'A' and 'B' matrices must be provided")
	})

	t.Run("invalid payload", func(t *testing.T) {
		payload := "invalid"
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("performance measurement", func(t *testing.T) {
		payload := map[string]interface{}{
			"size": 50,
		}

		start := time.Now()
		result, err := challenger.Execute(payload, log)
		elapsed := time.Since(start)

		assert.NoError(t, err)
		response, ok := result.(map[string]interface{})
		require.True(t, ok)

		// Verify computation time is reasonable
		compTime, ok := response["computationTimeMs"].(float64)
		require.True(t, ok)
		assert.Greater(t, compTime, 0.0)
		// Convert elapsed to float64 milliseconds for proper comparison
		elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
		assert.Less(t, compTime, elapsedMs*2) // Should not be more than 2x total time
	})

	t.Run("precision parameter", func(t *testing.T) {
		payload := map[string]interface{}{
			"size":      5,
			"precision": "float64",
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)

		response, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Contains(t, response, "computationTimeMs")
	})
}

func TestMatrixMultiplicationChallenge_Performance(t *testing.T) {
	log := zap.NewNop()
	challenger := challengers.NewMatrixMultiplicationChallenger()

	// Test performance with different matrix sizes
	sizes := []int{50, 100, 200}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			payload := map[string]interface{}{
				"size": size,
			}
			
			start := time.Now()
			result, err := challenger.Execute(payload, log)
			totalTime := time.Since(start)
			
			assert.NoError(t, err)
			
			response, ok := result.(map[string]interface{})
			require.True(t, ok)
			
			compTime, ok := response["computationTimeMs"].(float64)
			require.True(t, ok)
			
			flops := response["flops"].(int64)
			gflops := float64(flops) / (compTime / 1000.0) / 1e9
			
			t.Logf("Size: %d, Computation: %.2fms, Total: %v, GFLOPS: %.2f",
				size, compTime, totalTime, gflops)
			
			// Verify computation time is reasonable
			assert.Greater(t, compTime, 0.0)
			// Convert totalTime to float64 milliseconds for proper comparison
			totalTimeMs := float64(totalTime.Nanoseconds()) / 1e6
			assert.Less(t, compTime, totalTimeMs*1.5)
		})
	}
}

func TestMatrixMultiplicationChallenge_Accuracy(t *testing.T) {
	log := zap.NewNop()
	challenger := challengers.NewMatrixMultiplicationChallenger()

	// Test with known mathematical properties
	t.Run("associativity", func(t *testing.T) {
		// (A*B)*C should equal A*(B*C)
		A := [][]float64{{1, 2}, {3, 4}}
		B := [][]float64{{5, 6}, {7, 8}}
		C := [][]float64{{9, 10}, {11, 12}}
		
		// Calculate (A*B)*C
		payload1 := map[string]interface{}{
			"A": A,
			"B": B,
		}
		result1, err := challenger.Execute(payload1, log)
		assert.NoError(t, err)
		
		AB := result1.(map[string]interface{})["C"].([][]float64)
		
		payload2 := map[string]interface{}{
			"A": AB,
			"B": C,
		}
		result2, err := challenger.Execute(payload2, log)
		assert.NoError(t, err)
		
		AB_C := result2.(map[string]interface{})["C"].([][]float64)
		
		// Calculate A*(B*C)
		payload3 := map[string]interface{}{
			"A": B,
			"B": C,
		}
		result3, err := challenger.Execute(payload3, log)
		assert.NoError(t, err)
		
		BC := result3.(map[string]interface{})["C"].([][]float64)
		
		payload4 := map[string]interface{}{
			"A": A,
			"B": BC,
		}
		result4, err := challenger.Execute(payload4, log)
		assert.NoError(t, err)
		
		A_BC := result4.(map[string]interface{})["C"].([][]float64)
		
		// Compare results
		for i := range AB_C {
			for j := range AB_C[i] {
				assert.InDelta(t, AB_C[i][j], A_BC[i][j], 1e-10)
			}
		}
	})
}

func TestMatrixMultiplicationChallenge_WithGPUBackend(t *testing.T) {
	// This test verifies integration with the GPU backend
	slogger := slog.Default()
	gpuManager, err := gpu.NewManager(slogger)
	if err != nil {
		t.Skip("GPU manager not available:", err)
	}
	
	if !gpuManager.IsGPUAvailable() {
		t.Skip("GPU backend not available")
	}
	
	defer gpuManager.Cleanup()
	
	// Test matrix multiplication through GPU backend
	size := 100
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	
	// Initialize matrices
	for i := range a {
		a[i] = float32(i%10) / 10.0
		b[i] = float32((i+1)%10) / 10.0
	}
	
	// Perform multiplication
	start := time.Now()
	result, err := gpuManager.MatrixMultiply(a, b, size, size, size)
	elapsed := time.Since(start)
	
	assert.NoError(t, err)
	assert.Equal(t, size*size, len(result))
	
	// Calculate performance
	flops := float64(2 * size * size * size)
	gflops := flops / elapsed.Seconds() / 1e9
	
	t.Logf("GPU Matrix multiplication: size=%d, time=%v, GFLOPS=%.2f", size, elapsed, gflops)
	
	// Verify result is reasonable (not all zeros, not all NaN/Inf)
	var sum float32
	for _, val := range result {
		assert.False(t, math.IsNaN(float64(val)))
		assert.False(t, math.IsInf(float64(val), 0))
		sum += val
	}
	assert.NotEqual(t, float32(0), sum)
}