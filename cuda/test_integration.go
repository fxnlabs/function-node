//go:build cuda
// +build cuda

package cuda

import (
	"fmt"
	"log/slog"
	"testing"

	"github.com/fxnlabs/function-node/internal/gpu"
)

func TestCUDAIntegration(t *testing.T) {
	// Create a logger
	logger := slog.Default()

	// Create GPU manager
	manager, err := gpu.NewManager(logger)
	if err != nil {
		t.Fatalf("Failed to create GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Get device info
	info := manager.GetDeviceInfo()
	fmt.Println("=== CUDA Device Information ===")
	fmt.Printf("Backend Type: %s\n", manager.GetBackendType())
	fmt.Printf("Device Name: %s\n", info.Name)
	fmt.Printf("Compute Capability: %s\n", info.ComputeCapability)
	fmt.Printf("Total Memory: %d MB\n", info.TotalMemory/(1024*1024))
	fmt.Printf("Available Memory: %d MB\n", info.AvailableMemory/(1024*1024))
	if info.CUDAVersion != "" {
		fmt.Printf("CUDA Version: %s\n", info.CUDAVersion)
		fmt.Printf("Driver Version: %s\n", info.DriverVersion)
	}

	// Test matrix multiplication
	fmt.Println("\n=== Testing Matrix Multiplication ===")
	
	// Create test matrices (3x3)
	a := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	b := []float32{
		9, 8, 7,
		6, 5, 4,
		3, 2, 1,
	}

	// Perform multiplication
	result, err := manager.MatrixMultiply(a, b, 3, 3, 3)
	if err != nil {
		t.Fatalf("Matrix multiplication failed: %v", err)
	}

	// Print result
	fmt.Println("Result:")
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%6.2f ", result[i*3+j])
		}
		fmt.Println()
	}

	// Expected result:
	// [30  24  18]
	// [84  69  54]
	// [138 114 90]

	fmt.Println("\nCUDA integration test completed successfully!")
}