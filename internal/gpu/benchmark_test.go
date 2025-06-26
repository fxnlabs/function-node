package gpu

import (
	"fmt"
	"log/slog"
	"testing"
)

func BenchmarkGPUBackend_MatrixMultiply(b *testing.B) {
	logger := slog.Default()
	backend := NewGPUBackend(logger)
	defer backend.Cleanup()
	
	sizes := []int{64, 128, 256, 512, 1024}
	
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
		})
	}
}

func BenchmarkCPUBackend_MatrixMultiply(b *testing.B) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	defer backend.Cleanup()
	
	// Use smaller sizes for CPU to keep benchmark reasonable
	sizes := []int{32, 64, 128, 256}
	
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
			b.ReportMetric(float64(size*size*4*3)/(1<<20), "MB")
		})
	}
}

// Benchmark memory allocation overhead
func BenchmarkGPUBackend_MemoryOverhead(b *testing.B) {
	logger := slog.Default()
	backend := NewGPUBackend(logger)
	defer backend.Cleanup()
	
	size := 256
	
	b.Run("with_allocation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// Create new matrices each time
			a := make([]float32, size*size)
			bb := make([]float32, size*size)
			
			for j := range a {
				a[j] = float32(j) / float32(size)
				bb[j] = float32(j+1) / float32(size)
			}
			
			_, err := backend.MatrixMultiply(a, bb, size, size, size)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	
	b.Run("reuse_memory", func(b *testing.B) {
		// Pre-allocate matrices
		a := make([]float32, size*size)
		bb := make([]float32, size*size)
		
		for j := range a {
			a[j] = float32(j) / float32(size)
			bb[j] = float32(j+1) / float32(size)
		}
		
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			_, err := backend.MatrixMultiply(a, bb, size, size, size)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}