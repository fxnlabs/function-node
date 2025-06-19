//go:build !cuda
// +build !cuda

package gpu

// tryCreateCUDABackend attempts to create a CUDA backend when cuda build tag is NOT present
func (m *Manager) tryCreateCUDABackend() GPUBackend {
	return nil
}