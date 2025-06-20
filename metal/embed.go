//go:build metal && darwin
// +build metal,darwin

// Package metal provides embedded Metal shader libraries for GPU acceleration
package metal

import _ "embed"

// MetalLib contains the pre-compiled Metal shader library for matrix multiplication
//
//go:embed lib/matmul.metallib
var MetalLib []byte

// GetMetalLib returns the embedded Metal shader library data
func GetMetalLib() []byte {
	return MetalLib
}

// GetMetalLibSize returns the size of the embedded Metal shader library
func GetMetalLibSize() int {
	return len(MetalLib)
}