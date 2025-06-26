package challenge

import (
	"crypto/sha256"
	"fmt"
	"math/rand"
)

// FreivaldsVerify performs Freivalds' algorithm to probabilistically verify that C = A * B
// Returns true if the multiplication is likely correct, false otherwise
// The algorithm has a false positive rate of at most 1/2^k where k is the number of iterations
func FreivaldsVerify(A, B, C [][]float64, iterations int) bool {
	if len(A) == 0 || len(B) == 0 || len(C) == 0 {
		return false
	}

	n := len(A)
	m := len(A[0])
	p := len(B[0])

	// Verify dimensions
	if len(B) != m || len(C) != n || len(C[0]) != p {
		return false
	}

	// Perform k iterations of Freivalds' test
	for i := 0; i < iterations; i++ {
		// Generate random vector r
		r := make([]float64, p)
		for j := 0; j < p; j++ {
			r[j] = float64(rand.Intn(2)) // Binary vector for simplicity
		}

		// Compute Br
		Br := multiplyMatrixVector(B, r)

		// Compute A(Br)
		ABr := multiplyMatrixVector(A, Br)

		// Compute Cr
		Cr := multiplyMatrixVector(C, r)

		// Check if A(Br) == Cr
		if !vectorsEqual(ABr, Cr, 1e-9) {
			return false
		}
	}

	return true
}

// multiplyMatrixVector multiplies a matrix by a vector
func multiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	result := make([]float64, rows)

	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += matrix[i][j] * vector[j]
		}
		result[i] = sum
	}

	return result
}

// vectorsEqual checks if two vectors are equal within a tolerance
func vectorsEqual(a, b []float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			return false
		}
	}

	return true
}

// VerificationData represents the verification information for a matrix multiplication result
type VerificationData struct {
	MerkleRoot    string                   `json:"merkleRoot"`
	ResultSamples []map[string]interface{} `json:"resultSamples"`
}

// GenerateVerificationData creates verification data for a matrix result
func GenerateVerificationData(matrix [][]float64, sampleCount int) VerificationData {
	return VerificationData{
		MerkleRoot:    GenerateMerkleRoot(matrix),
		ResultSamples: GenerateResultSamples(matrix, sampleCount),
	}
}

// GenerateMerkleRoot generates a merkle root from a matrix
func GenerateMerkleRoot(matrix [][]float64) string {
	// Simplified merkle root generation
	// In production, this would build a proper merkle tree
	var data []byte
	for i := range matrix {
		for j := range matrix[i] {
			data = append(data, []byte(fmt.Sprintf("%.6f", matrix[i][j]))...)
		}
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("0x%x", hash)
}

// GenerateResultSamples returns sample values from a matrix
func GenerateResultSamples(matrix [][]float64, count int) []map[string]interface{} {
	samples := make([]map[string]interface{}, 0, count)
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return samples
	}

	rows := len(matrix)
	cols := len(matrix[0])

	// Sample at specific positions
	positions := [][]int{
		{0, 0},               // First element
		{rows / 2, cols / 2}, // Middle element
		{rows - 1, cols - 1}, // Last element
		{rows / 4, cols / 4}, // Quarter position
		{3 * rows / 4, 3 * cols / 4}, // Three-quarter position
	}

	for i := 0; i < count && i < len(positions); i++ {
		row := positions[i][0]
		col := positions[i][1]
		if row < rows && col < cols {
			samples = append(samples, map[string]interface{}{
				"row":   row,
				"col":   col,
				"value": matrix[row][col],
			})
		}
	}

	return samples
}