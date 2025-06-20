//go:build !metal || !darwin
// +build !metal !darwin

package gpu

import "log/slog"

// MetalBackend is a stub type when Metal is not available
type MetalBackend struct {
	logger *slog.Logger
}

// Stub implementations to satisfy GPUBackend interface
func (m *MetalBackend) MatrixMultiply(a, b []float32, mDim, k, n int) ([]float32, error) {
	panic("Metal backend not available")
}

func (m *MetalBackend) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{Name: "Metal not available"}
}

func (m *MetalBackend) IsAvailable() bool {
	return false
}

func (m *MetalBackend) Initialize() error {
	panic("Metal backend not available")
}

func (m *MetalBackend) Cleanup() error {
	return nil
}