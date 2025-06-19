// +build cuda

package gpu

import (
	"fmt"
	"log/slog"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCUDABackend_Initialize(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	// Test initialization
	err := backend.Initialize()
	assert.NoError(t, err)
	assert.True(t, backend.initialized)
	
	// Test device info
	info := backend.GetDeviceInfo()
	assert.NotEmpty(t, info.Name)
	assert.Greater(t, info.TotalMemory, int64(0))
	assert.NotEmpty(t, info.ComputeCapability)
	
	// Test double initialization (should be idempotent)
	err = backend.Initialize()
	assert.NoError(t, err)
	
	// Cleanup
	err = backend.Cleanup()
	assert.NoError(t, err)
	assert.False(t, backend.initialized)
}

func TestCUDABackend_MatrixMultiply(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	defer backend.Cleanup()
	
	testCases := []struct {
		name        string
		m, k, n     int
		setupA      func([]float32)
		setupB      func([]float32)
		verifyC     func(*testing.T, []float32)
		expectError bool
	}{
		{
			name: "small identity matrices",
			m: 3, k: 3, n: 3,
			setupA: func(a []float32) {
				// Identity matrix
				for i := 0; i < 3; i++ {
					a[i*3+i] = 1.0
				}
			},
			setupB: func(b []float32) {
				// Identity matrix
				for i := 0; i < 3; i++ {
					b[i*3+i] = 1.0
				}
			},
			verifyC: func(t *testing.T, c []float32) {
				// Result should be identity
				for i := 0; i < 3; i++ {
					for j := 0; j < 3; j++ {
						expected := float32(0.0)
						if i == j {
							expected = 1.0
						}
						assert.InDelta(t, expected, c[i*3+j], 1e-5)
					}
				}
			},
		},
		{
			name: "simple 2x2 multiplication",
			m: 2, k: 2, n: 2,
			setupA: func(a []float32) {
				a[0], a[1] = 1, 2
				a[2], a[3] = 3, 4
			},
			setupB: func(b []float32) {
				b[0], b[1] = 5, 6
				b[2], b[3] = 7, 8
			},
			verifyC: func(t *testing.T, c []float32) {
				// Expected: [[19, 22], [43, 50]]
				assert.InDelta(t, float32(19), c[0], 1e-5)
				assert.InDelta(t, float32(22), c[1], 1e-5)
				assert.InDelta(t, float32(43), c[2], 1e-5)
				assert.InDelta(t, float32(50), c[3], 1e-5)
			},
		},
		{
			name: "rectangular matrices",
			m: 2, k: 3, n: 4,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = float32(i + 1)
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = float32(i + 1)
				}
			},
			verifyC: func(t *testing.T, c []float32) {
				// Verify dimensions are correct
				assert.Equal(t, 8, len(c)) // 2x4
			},
		},
		{
			name: "large matrices for performance",
			m: 512, k: 512, n: 512,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = float32(i%10) / 10.0
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = float32(i%10) / 10.0
				}
			},
			verifyC: func(t *testing.T, c []float32) {
				// Just verify size and some basic properties
				assert.Equal(t, 512*512, len(c))
				// All values should be finite
				for _, val := range c {
					assert.False(t, math.IsNaN(float64(val)))
					assert.False(t, math.IsInf(float64(val), 0))
				}
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Setup matrices
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.k*tc.n)
			
			if tc.setupA != nil {
				tc.setupA(a)
			}
			if tc.setupB != nil {
				tc.setupB(b)
			}
			
			// Perform multiplication
			result, err := backend.MatrixMultiply(a, b, tc.m, tc.k, tc.n)
			
			if tc.expectError {
				assert.Error(t, err)
				return
			}
			
			require.NoError(t, err)
			assert.Equal(t, tc.m*tc.n, len(result))
			
			if tc.verifyC != nil {
				tc.verifyC(t, result)
			}
		})
	}

	// Test dimension mismatch cases separately
	t.Run("dimension mismatch - wrong A size", func(t *testing.T) {
		// Create array A with wrong size
		a := make([]float32, 5) // Should be 6 (2*3)
		b := make([]float32, 6) // Correct size (3*2)
		
		_, err := backend.MatrixMultiply(a, b, 2, 3, 2)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "matrix A size mismatch")
	})

	t.Run("dimension mismatch - wrong B size", func(t *testing.T) {
		// Create array B with wrong size
		a := make([]float32, 6) // Correct size (2*3)
		b := make([]float32, 5) // Should be 6 (3*2)
		
		_, err := backend.MatrixMultiply(a, b, 2, 3, 2)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "matrix B size mismatch")
	})
}

func TestCUDABackend_Performance(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	defer backend.Cleanup()
	
	// Test different matrix sizes and measure performance
	sizes := []int{128, 256, 512, 1024}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create matrices
			a := make([]float32, size*size)
			b := make([]float32, size*size)
			
			// Initialize with random values
			for i := range a {
				a[i] = float32(i%100) / 100.0
				b[i] = float32((i+1)%100) / 100.0
			}
			
			// Warm up
			_, err := backend.MatrixMultiply(a, b, size, size, size)
			require.NoError(t, err)
			
			// Measure performance
			start := time.Now()
			result, err := backend.MatrixMultiply(a, b, size, size, size)
			elapsed := time.Since(start)
			
			require.NoError(t, err)
			assert.Equal(t, size*size, len(result))
			
			// Calculate GFLOPS
			flops := float64(2 * size * size * size)
			gflops := flops / elapsed.Seconds() / 1e9
			
			t.Logf("Matrix size: %dx%d, Time: %v, GFLOPS: %.2f", 
				size, size, elapsed, gflops)
			
			// Performance should be reasonable (at least 10 GFLOPS for GPU)
			if size >= 512 {
				assert.Greater(t, gflops, 10.0, "GPU performance too low")
			}
		})
	}
}

func TestCUDABackend_MemoryManagement(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	defer backend.Cleanup()
	
	// Test multiple operations to check for memory leaks
	size := 256
	iterations := 100
	
	for i := 0; i < iterations; i++ {
		a := make([]float32, size*size)
		b := make([]float32, size*size)
		
		// Initialize
		for j := range a {
			a[j] = float32(j) / float32(size)
			b[j] = float32(j+1) / float32(size)
		}
		
		result, err := backend.MatrixMultiply(a, b, size, size, size)
		require.NoError(t, err)
		assert.Equal(t, size*size, len(result))
		
		// Verify at least one element to ensure computation happened
		assert.NotEqual(t, float32(0), result[0])
	}
}

func TestCUDABackend_EdgeCases(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	defer backend.Cleanup()
	
	
	t.Run("single element", func(t *testing.T) {
		a := []float32{2.0}
		b := []float32{3.0}
		result, err := backend.MatrixMultiply(a, b, 1, 1, 1)
		require.NoError(t, err)
		assert.InDelta(t, float32(6.0), result[0], 1e-5)
	})
	
	t.Run("very small values", func(t *testing.T) {
		a := []float32{1e-10, 1e-10, 1e-10, 1e-10}
		b := []float32{1e-10, 1e-10, 1e-10, 1e-10}
		result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		// Should handle small values without underflow to zero
		for _, val := range result {
			assert.Greater(t, val, float32(0))
		}
	})
	
	t.Run("very large values", func(t *testing.T) {
		a := []float32{1e10, 1e10, 1e10, 1e10}
		b := []float32{1e-10, 1e-10, 1e-10, 1e-10}
		result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		// Should handle without overflow
		for _, val := range result {
			assert.False(t, math.IsInf(float64(val), 0))
			assert.False(t, math.IsNaN(float64(val)))
		}
	})
}

func TestCUDABackend_Cleanup(t *testing.T) {
	logger := slog.Default()
	backend := NewCUDABackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("CUDA not available on this system")
	}
	
	// Initialize
	err := backend.Initialize()
	require.NoError(t, err)
	
	// Use it
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	_, err = backend.MatrixMultiply(a, b, 2, 2, 2)
	require.NoError(t, err)
	
	// Cleanup
	err = backend.Cleanup()
	assert.NoError(t, err)
	
	// Double cleanup should be safe
	err = backend.Cleanup()
	assert.NoError(t, err)
	
	// Should auto-initialize on next use
	_, err = backend.MatrixMultiply(a, b, 2, 2, 2)
	assert.NoError(t, err)
}