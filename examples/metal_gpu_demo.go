//go:build metal && darwin
// +build metal,darwin

package main

import (
	"fmt"
	"log"
	"log/slog"
	"time"

	"github.com/fxnlabs/function-node/internal/gpu"
)

func main() {
	// Initialize logger
	logger := slog.Default()

	// Create GPU manager
	manager, err := gpu.NewManager(logger)
	if err != nil {
		log.Fatalf("Failed to create GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Get device information
	deviceInfo := manager.GetDeviceInfo()
	fmt.Printf("GPU Backend: %s\n", manager.GetBackendType())
	fmt.Printf("Device: %s\n", deviceInfo.Name)
	fmt.Printf("Compute Capability: %s\n", deviceInfo.ComputeCapability)
	fmt.Printf("Total Memory: %.2f GB\n", float64(deviceInfo.TotalMemory)/(1<<30))
	fmt.Printf("Is GPU Available: %v\n", manager.IsGPUAvailable())
	fmt.Println()

	// Test matrix multiplication with different sizes
	sizes := []int{64, 128, 256, 512, 1024}
	
	for _, size := range sizes {
		fmt.Printf("Testing %dx%d matrix multiplication...\n", size, size)
		
		// Create random matrices
		a := make([]float32, size*size)
		b := make([]float32, size*size)
		
		// Initialize with random values
		for i := range a {
			a[i] = float32(i%100) / 100.0
			b[i] = float32((i+1)%100) / 100.0
		}
		
		// Perform matrix multiplication
		start := time.Now()
		result, err := manager.MatrixMultiply(a, b, size, size, size)
		elapsed := time.Since(start)
		
		if err != nil {
			log.Printf("Matrix multiplication failed: %v", err)
			continue
		}
		
		// Calculate performance metrics
		flops := int64(2 * size * size * size)
		gflops := float64(flops) / elapsed.Seconds() / 1e9
		
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Performance: %.2f GFLOPS\n", gflops)
		fmt.Printf("  Result size: %d elements\n", len(result))
		
		// Verify result (spot check)
		if size <= 4 {
			fmt.Printf("  Result[0,0]: %.2f\n", result[0])
		}
		
		fmt.Println()
	}
	
	// Demonstrate backend info
	fmt.Printf("Backend type: %s\n", manager.GetBackendType())
	if manager.IsGPUAvailable() {
		fmt.Println("GPU acceleration is active!")
	} else {
		fmt.Println("Running on CPU fallback")
	}
}