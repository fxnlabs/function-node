//go:build !cuda && !metal
// +build !cuda,!metal

package gpu

import (
	"log/slog"
)

// NewGPUBackend creates an appropriate GPU backend based on available hardware
// Without GPU support, it will always return CPU backend
func NewGPUBackend(logger *slog.Logger) GPUBackend {
	// Only CPU backend available
	logger.Info("Using CPU backend (compiled without GPU support)")
	return NewCPUBackend(logger)
}