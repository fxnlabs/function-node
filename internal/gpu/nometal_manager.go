//go:build !metal || !darwin
// +build !metal !darwin

package gpu

// tryCreateMetalBackend attempts to create a Metal backend when metal build tag is NOT present
func (m *Manager) tryCreateMetalBackend() GPUBackend {
	return nil
}