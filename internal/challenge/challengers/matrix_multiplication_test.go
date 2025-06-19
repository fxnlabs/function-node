package challengers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestMatrixMultiplicationChallenger_Execute(t *testing.T) {
	log := zap.NewNop()
	challenger := NewMatrixMultiplicationChallenger()

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

	t.Run("invalid matrix A", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": "invalid",
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid matrix B", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
			"B": "invalid",
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("missing matrix A", func(t *testing.T) {
		payload := map[string]interface{}{
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("missing matrix B", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

/*
// This test is no longer relevant with the new GPU manager architecture
func TestMatrixMultiplicationChallenger_detectGPU(t *testing.T) {
	challenger := &MatrixMultiplicationChallenger{}
	challenger.detectGPU()

	// We can't assert specific values since it depends on the system
	// But we can verify the fields are set
	assert.NotEmpty(t, challenger.gpuInfo)
	// gpuAvailable will be true or false depending on system
}
*/

func TestGenerateRandomMatrix(t *testing.T) {
	rows, cols := 5, 7
	matrix := generateRandomMatrix(rows, cols)

	assert.Equal(t, rows, len(matrix))
	for i := range matrix {
		assert.Equal(t, cols, len(matrix[i]))
		for j := range matrix[i] {
			// Verify values are in [0, 1) range
			assert.GreaterOrEqual(t, matrix[i][j], 0.0)
			assert.Less(t, matrix[i][j], 1.0)
		}
	}
}

func TestMatrixMultiplicationChallenger_multiplyMatricesCPU(t *testing.T) {
	challenger := NewMatrixMultiplicationChallenger()

	testCases := []struct {
		name     string
		matrixA  [][]float64
		matrixB  [][]float64
		expected [][]float64
	}{
		{
			name:     "2x2 matrices",
			matrixA:  [][]float64{{1, 2}, {3, 4}},
			matrixB:  [][]float64{{5, 6}, {7, 8}},
			expected: [][]float64{{19, 22}, {43, 50}},
		},
		{
			name:     "identity matrices",
			matrixA:  [][]float64{{1, 0}, {0, 1}},
			matrixB:  [][]float64{{5, 6}, {7, 8}},
			expected: [][]float64{{5, 6}, {7, 8}},
		},
		{
			name:     "rectangular matrices",
			matrixA:  [][]float64{{1, 2, 3}, {4, 5, 6}}, // 2x3
			matrixB:  [][]float64{{7, 8}, {9, 10}, {11, 12}}, // 3x2
			expected: [][]float64{{58, 64}, {139, 154}}, // 2x2
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := challenger.multiplyMatricesCPU(tc.matrixA, tc.matrixB)
			assert.NoError(t, err)
			
			// Compare with expected result
			require.Equal(t, len(tc.expected), len(result))
			for i := range tc.expected {
				require.Equal(t, len(tc.expected[i]), len(result[i]))
				for j := range tc.expected[i] {
					assert.InDelta(t, tc.expected[i][j], result[i][j], 1e-10)
				}
			}
		})
	}

	t.Run("incompatible dimensions", func(t *testing.T) {
		matrixA := [][]float64{{1, 2}, {3, 4}} // 2x2
		matrixB := [][]float64{{5, 6, 7}}      // 1x3
		
		_, err := challenger.multiplyMatricesCPU(matrixA, matrixB)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not compatible for multiplication")
	})
}

/*
// This test is no longer relevant with the new GPU manager architecture
func TestMatrixMultiplicationChallenger_determineBackend(t *testing.T) {
	log := zap.NewNop()
	challenger := NewMatrixMultiplicationChallenger()

	testCases := []struct {
		name          string
		requested     string
		gpuAvailable  bool
		expectedCPU   bool // true if we expect CPU backend
	}{
		{"auto with GPU", "auto", true, false},
		{"auto without GPU", "auto", false, true},
		{"explicit CPU", "cpu", true, true},
		{"explicit CPU", "cpu", false, true},
		{"GPU with GPU", "gpu", true, false},
		{"GPU without GPU", "gpu", false, true},
		{"unknown backend", "unknown", true, false},
		{"unknown backend no GPU", "unknown", false, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Mock GPU availability
			challenger.gpuAvailable = tc.gpuAvailable
			
			backend := challenger.determineBackend(tc.requested, log)
			
			if tc.expectedCPU {
				assert.Equal(t, "cpu", backend)
			} else {
				assert.Equal(t, "gpu", backend)
			}
		})
	}
}
*/
