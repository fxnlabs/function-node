//go:build metal && darwin
// +build metal,darwin

package gpu

import (
	"log/slog"
)

// NewGPUBackend creates an appropriate GPU backend based on available hardware
// On macOS with Metal support, it will try Metal first, then fall back to CPU
func NewGPUBackend(logger *slog.Logger) GPUBackend {
	// Try Metal backend first on macOS
	metalBackend := NewMetalBackend(logger)
	if metalBackend.IsAvailable() {
		logger.Info("Using Metal GPU backend")
		return metalBackend
	}
	
	// Fall back to CPU
	logger.Info("Using CPU backend (no GPU available)")
	return NewCPUBackend(logger)
}