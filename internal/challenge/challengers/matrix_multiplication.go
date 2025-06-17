package challengers

import (
	"encoding/json"
	"fmt"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
)

// MatrixMultiplicationChallenger performs matrix multiplication.
type MatrixMultiplicationChallenger struct{}

// Execute performs a matrix multiplication challenge.
func (c *MatrixMultiplicationChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	log.Info("Performing matrix multiplication challenge...")

	data, err := json.Marshal(payload)
	if err != nil {
		log.Error("Failed to marshal payload", zap.Error(err))
		return nil, err
	}

	var matrices struct {
		A [][]float64 `json:"A"`
		B [][]float64 `json:"B"`
	}

	if err := json.Unmarshal(data, &matrices); err != nil {
		log.Error("Failed to unmarshal matrices from payload", zap.Error(err))
		return nil, err
	}

	if len(matrices.A) == 0 || len(matrices.B) == 0 {
		log.Error("Matrices A or B are empty")
		return nil, fmt.Errorf("matrices A or B are empty")
	}

	aRows, aCols := len(matrices.A), len(matrices.A[0])
	bRows, bCols := len(matrices.B), len(matrices.B[0])

	if aCols != bRows {
		log.Error("Matrix dimensions are not compatible for multiplication",
			zap.Int("a_cols", aCols),
			zap.Int("b_rows", bRows))
		return nil, fmt.Errorf("matrix dimensions are not compatible for multiplication")
	}

	a := mat.NewDense(aRows, aCols, nil)
	for i, row := range matrices.A {
		a.SetRow(i, row)
	}

	b := mat.NewDense(bRows, bCols, nil)
	for i, row := range matrices.B {
		b.SetRow(i, row)
	}

	var res mat.Dense
	res.Mul(a, b)

	r, cols := res.Dims()
	resultMatrix := make([][]float64, r)
	for i := range r {
		resultMatrix[i] = make([]float64, cols)
		for j := range cols {
			resultMatrix[i][j] = res.At(i, j)
		}
	}

	log.Info("Matrix multiplication successful")

	return map[string]interface{}{
		"C": resultMatrix,
	}, nil
}
