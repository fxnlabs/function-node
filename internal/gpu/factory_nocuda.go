//go:build !cuda
// +build !cuda

package gpu

import (
	"log/slog"
)

// NewGPUBackend creates an appropriate GPU backend based on available hardware
// Without CUDA support, it will always return CPU backend
func NewGPUBackend(logger *slog.Logger) GPUBackend {
	// Only CPU backend available
	logger.Info("Using CPU backend (compiled without CUDA support)")
	return NewCPUBackend(logger)
}