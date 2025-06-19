package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Helper function to create random matrix
func randomMatrix(rows, cols int) []float32 {
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = rand.Float32()
	}
	return matrix
}

// CPU matrix multiplication for verification
func matmulCPU(A, B []float32, M, N, K int) []float32 {
	C := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}
	return C
}

// Check if results match within tolerance
func compareResults(gpu, cpu []float32, tolerance float32) bool {
	if len(gpu) != len(cpu) {
		return false
	}
	for i := range gpu {
		diff := float32(math.Abs(float64(gpu[i] - cpu[i])))
		if diff > tolerance {
			return false
		}
	}
	return true
}

func TestMatMul(t *testing.T) {
	// Initialize CUDA
	if err := Init(); err != nil {
		t.Skipf("CUDA initialization failed: %v", err)
	}
	defer Cleanup()

	// Check device
	if err := CheckDevice(); err != nil {
		t.Skipf("No suitable CUDA device: %v", err)
	}

	// Get device info
	info, err := GetDeviceInfo()
	if err != nil {
		t.Logf("Could not get device info: %v", err)
	} else {
		t.Logf("Using CUDA device: %s (Compute Capability %d.%d)", 
			info.Name, info.Major, info.Minor)
	}

	// Test cases
	testCases := []struct {
		name string
		M, N, K int
	}{
		{"Small Square", 64, 64, 64},
		{"Medium Square", 256, 256, 256},
		{"Large Square", 512, 512, 512},
		{"Rectangular 1", 128, 256, 64},
		{"Rectangular 2", 256, 128, 512},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create random matrices
			A := randomMatrix(tc.M, tc.K)
			B := randomMatrix(tc.K, tc.N)

			// GPU computation
			start := time.Now()
			C_gpu, err := MatMul(A, B, tc.M, tc.N, tc.K)
			gpuTime := time.Since(start)

			if err != nil {
				t.Fatalf("GPU MatMul failed: %v", err)
			}

			t.Logf("GPU time for %dx%dx%d: %v", tc.M, tc.N, tc.K, gpuTime)

			// For smaller matrices, verify against CPU
			if tc.M <= 256 && tc.N <= 256 && tc.K <= 256 {
				start = time.Now()
				C_cpu := matmulCPU(A, B, tc.M, tc.N, tc.K)
				cpuTime := time.Since(start)

				t.Logf("CPU time: %v (speedup: %.2fx)", cpuTime, 
					float64(cpuTime)/float64(gpuTime))

				if !compareResults(C_gpu, C_cpu, 1e-3) {
					t.Errorf("Results don't match between GPU and CPU")
				}
			}
		})
	}
}

func BenchmarkMatMul(b *testing.B) {
	// Initialize CUDA
	if err := Init(); err != nil {
		b.Skipf("CUDA initialization failed: %v", err)
	}
	defer Cleanup()

	sizes := []int{256, 512, 1024, 2048}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			// Create matrices
			A := randomMatrix(size, size)
			B := randomMatrix(size, size)

			// Warm-up
			_, _ = MatMul(A, B, size, size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := MatMul(A, B, size, size, size)
				if err != nil {
					b.Fatalf("MatMul failed: %v", err)
				}
			}

			// Calculate GFLOPS
			flops := 2.0 * float64(size) * float64(size) * float64(size)
			gflops := (flops * float64(b.N)) / (float64(b.Elapsed()) * 1e9)
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}