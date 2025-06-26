//go:build cuda
// +build cuda

package gpu

// tryCreateCUDABackend attempts to create a CUDA backend when cuda build tag is present
func (m *Manager) tryCreateCUDABackend() GPUBackend {
	return NewCUDABackend(m.logger)
}