//go:build cuda
// +build cuda

package gpu

import (
	"log/slog"
)

// NewGPUBackend creates an appropriate GPU backend based on available hardware
// It will try CUDA first, then fall back to CPU
func NewGPUBackend(logger *slog.Logger) GPUBackend {
	// Try CUDA backend first
	cudaBackend := NewCUDABackend(logger)
	if cudaBackend.IsAvailable() {
		logger.Info("Using CUDA GPU backend")
		return cudaBackend
	}
	
	// Fall back to CPU
	logger.Info("Using CPU backend (no GPU available)")
	return NewCPUBackend(logger)
}