package gpu

// Float64ToFloat32 converts a slice of float64 to float32
func Float64ToFloat32(input []float64) []float32 {
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(v)
	}
	return output
}

// Float32ToFloat64 converts a slice of float32 to float64
func Float32ToFloat64(input []float32) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

// Float64MatrixToFloat32 converts a 2D float64 matrix to a flat float32 array in row-major order
func Float64MatrixToFloat32(matrix [][]float64) []float32 {
	if len(matrix) == 0 {
		return []float32{}
	}
	
	rows := len(matrix)
	cols := len(matrix[0])
	result := make([]float32, rows*cols)
	
	idx := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[idx] = float32(matrix[i][j])
			idx++
		}
	}
	
	return result
}

// Float32ArrayToFloat64Matrix converts a flat float32 array to a 2D float64 matrix
func Float32ArrayToFloat64Matrix(array []float32, rows, cols int) [][]float64 {
	if len(array) != rows*cols {
		return nil
	}
	
	matrix := make([][]float64, rows)
	idx := 0
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = float64(array[idx])
			idx++
		}
	}
	
	return matrix
}