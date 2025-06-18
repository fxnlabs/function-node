package challengers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestMatrixMultiplicationChallenger_Execute(t *testing.T) {
	log := zap.NewNop()
	challenger := &MatrixMultiplicationChallenger{}

	t.Run("valid multiplication", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		expected := map[string]interface{}{
			"C": [][]float64{{19, 22}, {43, 50}},
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)
		assert.Equal(t, expected, result)
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
