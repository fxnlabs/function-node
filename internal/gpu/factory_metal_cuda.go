//go:build metal && cuda && darwin
// +build metal,cuda,darwin

package gpu

import (
	"log/slog"
)

// NewGPUBackend creates an appropriate GPU backend based on available hardware
// When both Metal and CUDA are available, prefer Metal on macOS
func NewGPUBackend(logger *slog.Logger) GPUBackend {
	// Try Metal backend first on macOS
	metalBackend := NewMetalBackend(logger)
	if metalBackend.IsAvailable() {
		logger.Info("Using Metal GPU backend")
		return metalBackend
	}
	
	// Try CUDA backend next
	cudaBackend := NewCUDABackend(logger)
	if cudaBackend.IsAvailable() {
		logger.Info("Using CUDA GPU backend")
		return cudaBackend
	}
	
	// Fall back to CPU
	logger.Info("Using CPU backend (no GPU available)")
	return NewCPUBackend(logger)
}