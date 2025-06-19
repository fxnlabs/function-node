//go:build metal && darwin
// +build metal,darwin

package gpu

import (
	"fmt"
	"log/slog"
	"testing"
)

func TestMetalBackend_Initialize(t *testing.T) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	info := backend.GetDeviceInfo()
	if info.Name == "" {
		t.Error("Device name should not be empty")
	}
	if info.TotalMemory <= 0 {
		t.Error("Total memory should be greater than 0")
	}
	if info.AvailableMemory <= 0 {
		t.Error("Available memory should be greater than 0")
	}
	
	// Log device capabilities
	t.Logf("Metal Device: %s", info.Name)
	t.Logf("GPU Family: %s", info.ComputeCapability)
	t.Logf("Total Memory: %.2f GB", float64(info.TotalMemory)/(1<<30))
	t.Logf("Available Memory: %.2f GB", float64(info.AvailableMemory)/(1<<30))
	t.Logf("MPS Support: %v", backend.useMPS)
}

func TestMetalBackend_MatrixMultiply(t *testing.T) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	// Test cases
	tests := []struct {
		name string
		m, k, n int
		expectMPS bool
	}{
		{"Small matrices", 4, 4, 4, false},
		{"Medium matrices", 64, 64, 64, false},
		{"Large matrices", 256, 256, 256, false},
		{"MPS threshold matrices", 512, 512, 512, true},
		{"Large MPS matrices", 1024, 1024, 1024, true},
		{"Non-square matrices", 128, 256, 64, false},
		{"Non-square large matrices", 512, 1024, 256, true},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test matrices
			a := make([]float32, tt.m*tt.k)
			b := make([]float32, tt.k*tt.n)
			
			// Initialize with simple values
			for i := range a {
				a[i] = float32(i % 10)
			}
			for i := range b {
				b[i] = float32(i % 10)
			}
			
			// Perform multiplication
			result, err := backend.MatrixMultiply(a, b, tt.m, tt.k, tt.n)
			if err != nil {
				t.Errorf("Matrix multiplication failed: %v", err)
			}
			
			// Verify result dimensions
			if len(result) != tt.m*tt.n {
				t.Errorf("Result size mismatch: expected %d, got %d", tt.m*tt.n, len(result))
			}
			
			// Spot check a few values (full verification would require CPU computation)
			if tt.m >= 2 && tt.k >= 2 && tt.n >= 2 {
				// Check that result is not all zeros
				allZero := true
				for _, v := range result {
					if v != 0 {
						allZero = false
						break
					}
				}
				if allZero {
					t.Error("Result matrix is all zeros, which is unexpected")
				}
			}
		})
	}
}

func TestMetalBackend_EdgeCases(t *testing.T) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	// Test 1x1 matrix
	a := []float32{2.0}
	b := []float32{3.0}
	result, err := backend.MatrixMultiply(a, b, 1, 1, 1)
	if err != nil {
		t.Errorf("1x1 matrix multiplication failed: %v", err)
	}
	if len(result) != 1 || result[0] != 6.0 {
		t.Errorf("1x1 matrix multiplication incorrect: expected 6.0, got %v", result)
	}
	
	// Test dimension mismatch
	_, err = backend.MatrixMultiply(a, b, 2, 1, 1)
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

func TestMetalBackend_MPSThreshold(t *testing.T) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	// Test default threshold
	if backend.mpsThreshold != 512 {
		t.Errorf("Default MPS threshold should be 512, got %d", backend.mpsThreshold)
	}
	
	// Test custom threshold
	backend.SetMPSThreshold(256)
	if backend.mpsThreshold != 256 {
		t.Errorf("MPS threshold should be 256 after setting, got %d", backend.mpsThreshold)
	}
	
	// Test matrix multiplication with custom threshold
	size := 256
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	
	for i := range a {
		a[i] = 1.0
		b[i] = 2.0
	}
	
	// This should use MPS with the new threshold
	result, err := backend.MatrixMultiply(a, b, size, size, size)
	if err != nil {
		t.Fatalf("Matrix multiplication failed: %v", err)
	}
	
	// Verify result (each element should be size * 2.0)
	expected := float32(size * 2.0)
	for i := 0; i < 10; i++ { // Check first 10 elements
		if result[i] != expected {
			t.Errorf("Result[%d] = %f, expected %f", i, result[i], expected)
		}
	}
}

func TestMetalBackend_VerifyMPSCorrectness(t *testing.T) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		t.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	// Compare results between custom kernel and MPS
	sizes := []int{256, 512, 768}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test matrices
			a := make([]float32, size*size)
			b := make([]float32, size*size)
			
			// Initialize with random values
			for i := range a {
				a[i] = float32(i%10) / 10.0
				b[i] = float32((i+1)%10) / 10.0
			}
			
			// Force custom kernel by setting high threshold
			backend.SetMPSThreshold(10000)
			resultCustom, err := backend.MatrixMultiply(a, b, size, size, size)
			if err != nil {
				t.Fatalf("Custom kernel multiplication failed: %v", err)
			}
			
			// Force MPS by setting low threshold
			backend.SetMPSThreshold(1)
			resultMPS, err := backend.MatrixMultiply(a, b, size, size, size)
			if err != nil {
				t.Fatalf("MPS multiplication failed: %v", err)
			}
			
			// Compare results (allowing for small floating-point differences)
			const epsilon = 1e-4
			diffCount := 0
			maxDiff := float32(0)
			
			for i := range resultCustom {
				diff := abs(resultCustom[i] - resultMPS[i])
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > epsilon {
					diffCount++
					if diffCount < 5 { // Log first few differences
						t.Logf("Difference at [%d]: custom=%f, mps=%f, diff=%f", 
							i, resultCustom[i], resultMPS[i], diff)
					}
				}
			}
			
			if diffCount > 0 {
				t.Errorf("Found %d differences > %f (max diff: %f) out of %d elements",
					diffCount, epsilon, maxDiff, len(resultCustom))
			} else {
				t.Logf("Results match within tolerance (max diff: %e)", maxDiff)
			}
		})
	}
	
	// Reset to default threshold
	backend.SetMPSThreshold(512)
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func BenchmarkMetalBackend_MatrixMultiply(b *testing.B) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		b.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		b.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	sizes := []int{64, 128, 256, 512, 1024, 2048}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			// Create matrices
			a := make([]float32, size*size)
			bb := make([]float32, size*size)
			
			// Initialize with values
			for i := range a {
				a[i] = float32(i%100) / 100.0
				bb[i] = float32((i+1)%100) / 100.0
			}
			
			// Warm up
			_, _ = backend.MatrixMultiply(a, bb, size, size, size)
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := backend.MatrixMultiply(a, bb, size, size, size)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			// Report metrics
			flops := int64(2 * size * size * size * b.N)
			seconds := b.Elapsed().Seconds()
			gflops := float64(flops) / seconds / 1e9
			
			b.ReportMetric(gflops, "GFLOPS")
			b.ReportMetric(float64(size*size*4*3)/(1<<20), "MB") // Memory for A, B, C matrices
			
			// Note whether MPS was used
			if size >= backend.mpsThreshold {
				b.Logf("Used MPS for size %d", size)
			} else {
				b.Logf("Used custom kernel for size %d", size)
			}
		})
	}
}

func BenchmarkMetalBackend_MPSComparison(b *testing.B) {
	logger := slog.Default()
	backend := NewMetalBackend(logger)
	
	if !backend.IsAvailable() {
		b.Skip("Metal backend not available on this system")
	}
	
	err := backend.Initialize()
	if err != nil {
		b.Fatalf("Failed to initialize Metal backend: %v", err)
	}
	defer backend.Cleanup()
	
	sizes := []int{512, 1024, 2048}
	
	for _, size := range sizes {
		// Benchmark custom kernel
		b.Run(fmt.Sprintf("custom_kernel_%d", size), func(b *testing.B) {
			backend.SetMPSThreshold(10000) // Force custom kernel
			
			a := make([]float32, size*size)
			bb := make([]float32, size*size)
			for i := range a {
				a[i] = float32(i%100) / 100.0
				bb[i] = float32((i+1)%100) / 100.0
			}
			
			_, _ = backend.MatrixMultiply(a, bb, size, size, size)
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := backend.MatrixMultiply(a, bb, size, size, size)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			flops := int64(2 * size * size * size * b.N)
			seconds := b.Elapsed().Seconds()
			gflops := float64(flops) / seconds / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
		
		// Benchmark MPS
		b.Run(fmt.Sprintf("mps_%d", size), func(b *testing.B) {
			backend.SetMPSThreshold(1) // Force MPS
			
			a := make([]float32, size*size)
			bb := make([]float32, size*size)
			for i := range a {
				a[i] = float32(i%100) / 100.0
				bb[i] = float32((i+1)%100) / 100.0
			}
			
			_, _ = backend.MatrixMultiply(a, bb, size, size, size)
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := backend.MatrixMultiply(a, bb, size, size, size)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			flops := int64(2 * size * size * size * b.N)
			seconds := b.Elapsed().Seconds()
			gflops := float64(flops) / seconds / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
	
	// Reset to default
	backend.SetMPSThreshold(512)
}