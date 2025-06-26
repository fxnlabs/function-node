//go:build metal && darwin
// +build metal,darwin

package gpu

// tryCreateMetalBackend attempts to create a Metal backend when metal build tag is present
func (m *Manager) tryCreateMetalBackend() GPUBackend {
	return NewMetalBackend(m.logger)
}