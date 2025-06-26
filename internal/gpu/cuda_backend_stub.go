//go:build !cuda
// +build !cuda

package gpu

import "log/slog"

// CUDABackend is a stub type when CUDA is not available
type CUDABackend struct {
	logger *slog.Logger
}

// Stub implementations to satisfy GPUBackend interface
func (c *CUDABackend) MatrixMultiply(a, b []float32, m, k, n int) ([]float32, error) {
	panic("CUDA backend not available")
}

func (c *CUDABackend) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{Name: "CUDA not available"}
}

func (c *CUDABackend) IsAvailable() bool {
	return false
}

func (c *CUDABackend) Initialize() error {
	panic("CUDA backend not available")
}

func (c *CUDABackend) Cleanup() error {
	return nil
}