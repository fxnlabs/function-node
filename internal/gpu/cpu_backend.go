package gpu

import (
	"fmt"
	"log/slog"
	"runtime"
)

// CPUBackend implements GPUBackend using CPU for fallback
type CPUBackend struct {
	logger      *slog.Logger
	initialized bool
}

// NewCPUBackend creates a new CPU backend instance
func NewCPUBackend(logger *slog.Logger) *CPUBackend {
	return &CPUBackend{
		logger: logger,
	}
}

// Initialize prepares the CPU backend for use
func (c *CPUBackend) Initialize() error {
	if c.initialized {
		return nil
	}
	c.initialized = true
	c.logger.Info("CPU backend initialized")
	return nil
}

// Cleanup releases any resources (none for CPU backend)
func (c *CPUBackend) Cleanup() error {
	c.initialized = false
	return nil
}

// IsAvailable checks if the backend is available (always true for CPU)
func (c *CPUBackend) IsAvailable() bool {
	return true
}

// GetDeviceInfo returns device information for CPU
func (c *CPUBackend) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{
		Name:              fmt.Sprintf("CPU (%s)", runtime.GOARCH),
		TotalMemory:       getTotalSystemMemory(),
		AvailableMemory:   getAvailableSystemMemory(),
		ComputeCapability: "N/A",
		DriverVersion:     runtime.Version(),
	}
}

// MatrixMultiply performs matrix multiplication using CPU
// Implements C = A * B where A is m×k, B is k×n, and C is m×n
func (c *CPUBackend) MatrixMultiply(a, b []float32, m, k, n int) ([]float32, error) {
	if !c.initialized {
		return nil, fmt.Errorf("CPU backend not initialized")
	}

	// Validate dimensions
	if len(a) != m*k {
		return nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", m*k, len(a))
	}
	if len(b) != k*n {
		return nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", k*n, len(b))
	}

	// Allocate result matrix
	result := make([]float32, m*n)

	// Perform naive matrix multiplication
	// This is not optimized but serves as a simple fallback
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[l*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return result, nil
}

// getTotalSystemMemory returns total system memory in bytes
func getTotalSystemMemory() int64 {
	// Return a default value for now
	// In a real implementation, this would query system memory
	return 8 * 1024 * 1024 * 1024 // 8GB
}

// getAvailableSystemMemory returns available system memory in bytes
func getAvailableSystemMemory() int64 {
	// Return a default value for now
	// In a real implementation, this would query available memory
	return 4 * 1024 * 1024 * 1024 // 4GB
}