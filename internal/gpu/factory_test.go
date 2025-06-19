package gpu

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewGPUBackend(t *testing.T) {
	logger := slog.Default()
	
	// Test factory function
	backend := NewGPUBackend(logger)
	
	// Should always return a valid backend
	assert.NotNil(t, backend)
	assert.True(t, backend.IsAvailable())
	
	// Test initialization
	err := backend.Initialize()
	assert.NoError(t, err)
	
	// Test device info
	info := backend.GetDeviceInfo()
	assert.NotEmpty(t, info.Name)
	
	// Cleanup
	err = backend.Cleanup()
	assert.NoError(t, err)
}

func TestGPUBackend_Fallback(t *testing.T) {
	logger := slog.Default()
	
	// In environments without CUDA, should fall back to CPU
	backend := NewGPUBackend(logger)
	
	// CPU backend is always available
	assert.True(t, backend.IsAvailable())
	
	// Initialize backend
	err := backend.Initialize()
	assert.NoError(t, err)
	defer backend.Cleanup()
	
	// Should be able to perform matrix multiplication
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	
	result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
	assert.NoError(t, err)
	assert.Equal(t, 4, len(result))
	
	// Verify computation is correct
	// [[1,2], [3,4]] * [[5,6], [7,8]] = [[19,22], [43,50]]
	assert.InDelta(t, float32(19), result[0], 1e-5)
	assert.InDelta(t, float32(22), result[1], 1e-5)
	assert.InDelta(t, float32(43), result[2], 1e-5)
	assert.InDelta(t, float32(50), result[3], 1e-5)
}