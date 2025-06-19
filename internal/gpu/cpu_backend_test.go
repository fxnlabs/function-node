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

func TestCPUBackend_Initialize(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	// CPU backend should always be available
	assert.True(t, backend.IsAvailable())
	
	// Test initialization
	err := backend.Initialize()
	assert.NoError(t, err)
	assert.True(t, backend.initialized)
	
	// Test device info
	info := backend.GetDeviceInfo()
	assert.Contains(t, info.Name, "CPU")
	assert.Greater(t, info.TotalMemory, int64(0))
	assert.Equal(t, "N/A", info.ComputeCapability)
	
	// Test double initialization (should be idempotent)
	err = backend.Initialize()
	assert.NoError(t, err)
	
	// Cleanup
	err = backend.Cleanup()
	assert.NoError(t, err)
	assert.False(t, backend.initialized)
}

func TestCPUBackend_MatrixMultiply(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	err := backend.Initialize()
	require.NoError(t, err)
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
				// A is 2x3
				a[0], a[1], a[2] = 1, 2, 3
				a[3], a[4], a[5] = 4, 5, 6
			},
			setupB: func(b []float32) {
				// B is 3x4
				for i := range b {
					b[i] = float32(i + 1)
				}
			},
			verifyC: func(t *testing.T, c []float32) {
				// C should be 2x4
				assert.Equal(t, 8, len(c))
				// Verify specific values
				// First row of C = [1,2,3] · [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
				// = [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
				// = [38, 44, 50, 56]
				assert.InDelta(t, float32(38), c[0], 1e-5)
				assert.InDelta(t, float32(44), c[1], 1e-5)
				assert.InDelta(t, float32(50), c[2], 1e-5)
				assert.InDelta(t, float32(56), c[3], 1e-5)
			},
		},
		{
			name: "zero matrices",
			m: 3, k: 3, n: 3,
			setupA: func(a []float32) {
				// All zeros (default)
			},
			setupB: func(b []float32) {
				// All zeros (default)
			},
			verifyC: func(t *testing.T, c []float32) {
				// Result should be all zeros
				for _, val := range c {
					assert.Equal(t, float32(0), val)
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

func TestCPUBackend_Performance(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	err := backend.Initialize()
	require.NoError(t, err)
	defer backend.Cleanup()
	
	// Test different matrix sizes and measure performance
	sizes := []int{64, 128, 256}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create matrices
			a := make([]float32, size*size)
			b := make([]float32, size*size)
			
			// Initialize with values
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
			
			// CPU performance varies widely, just ensure it completes
			assert.Greater(t, gflops, 0.001, "Performance too low")
		})
	}
}

func TestCPUBackend_Correctness(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	err := backend.Initialize()
	require.NoError(t, err)
	defer backend.Cleanup()
	
	// Test against known matrix multiplication results
	t.Run("associativity", func(t *testing.T) {
		// (A*B)*C should equal A*(B*C)
		a := []float32{1, 2, 3, 4}     // 2x2
		b := []float32{5, 6, 7, 8}     // 2x2
		c := []float32{9, 10, 11, 12}  // 2x2
		
		// Calculate (A*B)*C
		ab, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		ab_c, err := backend.MatrixMultiply(ab, c, 2, 2, 2)
		require.NoError(t, err)
		
		// Calculate A*(B*C)
		bc, err := backend.MatrixMultiply(b, c, 2, 2, 2)
		require.NoError(t, err)
		a_bc, err := backend.MatrixMultiply(a, bc, 2, 2, 2)
		require.NoError(t, err)
		
		// Compare results
		for i := range ab_c {
			assert.InDelta(t, ab_c[i], a_bc[i], 1e-5)
		}
	})
	
	t.Run("distributivity", func(t *testing.T) {
		// A*(B+C) should equal A*B + A*C
		a := []float32{1, 2, 3, 4}  // 2x2
		b := []float32{5, 6, 7, 8}  // 2x2
		c := []float32{1, 1, 1, 1}  // 2x2
		
		// Calculate B+C
		bc := make([]float32, 4)
		for i := range bc {
			bc[i] = b[i] + c[i]
		}
		
		// Calculate A*(B+C)
		a_bc, err := backend.MatrixMultiply(a, bc, 2, 2, 2)
		require.NoError(t, err)
		
		// Calculate A*B + A*C
		ab, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		ac, err := backend.MatrixMultiply(a, c, 2, 2, 2)
		require.NoError(t, err)
		
		// Add results
		ab_plus_ac := make([]float32, 4)
		for i := range ab_plus_ac {
			ab_plus_ac[i] = ab[i] + ac[i]
		}
		
		// Compare
		for i := range a_bc {
			assert.InDelta(t, a_bc[i], ab_plus_ac[i], 1e-5)
		}
	})
}

func TestCPUBackend_EdgeCases(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	err := backend.Initialize()
	require.NoError(t, err)
	defer backend.Cleanup()
	
	t.Run("single element", func(t *testing.T) {
		a := []float32{2.0}
		b := []float32{3.0}
		result, err := backend.MatrixMultiply(a, b, 1, 1, 1)
		require.NoError(t, err)
		assert.InDelta(t, float32(6.0), result[0], 1e-5)
	})
	
	t.Run("row vector times column vector", func(t *testing.T) {
		// 1x3 * 3x1 = 1x1
		a := []float32{1, 2, 3}     // 1x3
		b := []float32{4, 5, 6}     // 3x1
		result, err := backend.MatrixMultiply(a, b, 1, 3, 1)
		require.NoError(t, err)
		// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		assert.InDelta(t, float32(32), result[0], 1e-5)
	})
	
	t.Run("column vector times row vector", func(t *testing.T) {
		// 3x1 * 1x3 = 3x3
		a := []float32{1, 2, 3}     // 3x1
		b := []float32{4, 5, 6}     // 1x3
		result, err := backend.MatrixMultiply(a, b, 3, 1, 3)
		require.NoError(t, err)
		assert.Equal(t, 9, len(result))
		// Result should be [[4,5,6], [8,10,12], [12,15,18]]
		expected := []float32{4, 5, 6, 8, 10, 12, 12, 15, 18}
		for i := range expected {
			assert.InDelta(t, expected[i], result[i], 1e-5)
		}
	})
	
	t.Run("negative values", func(t *testing.T) {
		a := []float32{-1, 2, -3, 4}
		b := []float32{5, -6, -7, 8}
		result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		// Verify computation with negative values
		// [-1*5 + 2*-7, -1*-6 + 2*8] = [-5-14, 6+16] = [-19, 22]
		// [-3*5 + 4*-7, -3*-6 + 4*8] = [-15-28, 18+32] = [-43, 50]
		assert.InDelta(t, float32(-19), result[0], 1e-5)
		assert.InDelta(t, float32(22), result[1], 1e-5)
		assert.InDelta(t, float32(-43), result[2], 1e-5)
		assert.InDelta(t, float32(50), result[3], 1e-5)
	})
	
	t.Run("very small values", func(t *testing.T) {
		a := []float32{1e-10, 1e-10, 1e-10, 1e-10}
		b := []float32{1e10, 1e10, 1e10, 1e10}
		result, err := backend.MatrixMultiply(a, b, 2, 2, 2)
		require.NoError(t, err)
		// Should handle without underflow/overflow
		for _, val := range result {
			assert.False(t, math.IsInf(float64(val), 0))
			assert.False(t, math.IsNaN(float64(val)))
		}
	})
}

func TestCPUBackend_AutoInitialization(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	// Backend should not be initialized by default
	assert.False(t, backend.initialized)
	
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	_, err := backend.MatrixMultiply(a, b, 2, 2, 2)
	
	// Should return error because backend is not initialized
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not initialized")
	assert.False(t, backend.initialized)
}