package gpu

import (
	"fmt"
	"log/slog"
	"sync"
)

// Manager handles GPU backend selection and lifecycle
type Manager struct {
	backend GPUBackend
	mu      sync.RWMutex
	logger  *slog.Logger
}

// NewManager creates a new GPU manager and selects the best available backend
func NewManager(logger *slog.Logger) (*Manager, error) {
	if logger == nil {
		logger = slog.Default()
	}
	
	m := &Manager{
		logger: logger,
	}
	
	// Try to detect and initialize the best backend
	if err := m.detectAndInitialize(); err != nil {
		return nil, err
	}
	
	return m, nil
}

// detectAndInitialize detects available backends and initializes the best one
func (m *Manager) detectAndInitialize() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Try CUDA first (only if build tag is enabled)
	if cudaBackend := m.tryCreateCUDABackend(); cudaBackend != nil {
		if cudaBackend.IsAvailable() {
			if err := cudaBackend.Initialize(); err == nil {
				m.backend = cudaBackend
				return nil
			}
			// If initialization failed, try cleanup
			_ = cudaBackend.Cleanup()
		}
	}

	// Fall back to CPU
	cpuBackend := NewCPUBackend(m.logger)
	if err := cpuBackend.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize CPU backend: %w", err)
	}
	m.backend = cpuBackend
	return nil
}


// GetBackend returns the current backend
func (m *Manager) GetBackend() GPUBackend {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.backend
}

// MatrixMultiply performs matrix multiplication using the selected backend
func (mgr *Manager) MatrixMultiply(a, b []float32, m, k, n int) ([]float32, error) {
	backend := mgr.GetBackend()
	if backend == nil {
		return nil, fmt.Errorf("no backend available")
	}
	return backend.MatrixMultiply(a, b, m, k, n)
}

// GetDeviceInfo returns device information from the current backend
func (m *Manager) GetDeviceInfo() DeviceInfo {
	backend := m.GetBackend()
	if backend == nil {
		return DeviceInfo{Name: "No backend available"}
	}
	return backend.GetDeviceInfo()
}

// IsGPUAvailable returns true if a GPU backend is active
func (m *Manager) IsGPUAvailable() bool {
	backend := m.GetBackend()
	if backend == nil {
		return false
	}
	// Check if it's not the CPU backend
	_, isCPU := backend.(*CPUBackend)
	return !isCPU
}

// Cleanup releases resources held by the current backend
func (m *Manager) Cleanup() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.backend != nil {
		if err := m.backend.Cleanup(); err != nil {
			return err
		}
		m.backend = nil
	}
	return nil
}

// GetBackendType returns a string describing the current backend type
func (m *Manager) GetBackendType() string {
	backend := m.GetBackend()
	if backend == nil {
		return "none"
	}
	
	// Check if it's CPU backend
	if _, isCPU := backend.(*CPUBackend); isCPU {
		return "cpu"
	}
	
	// If not CPU and GPU is available, it must be CUDA
	if m.IsGPUAvailable() {
		return "cuda"
	}
	
	return "unknown"
}