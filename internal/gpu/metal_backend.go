//go:build metal && darwin
// +build metal,darwin

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../../metal/include -x objective-c -fobjc-arc
#cgo LDFLAGS: ${SRCDIR}/../../metal/lib/libmetal_backend.a -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "metal_backend.h"
*/
import "C"
import (
	"fmt"
	"log/slog"
	"runtime"
	"unsafe"
	
	"github.com/fxnlabs/function-node/metal"
)

// MetalBackend implements GPUBackend using Apple Metal
type MetalBackend struct {
	logger       *slog.Logger
	initialized  bool
	deviceInfo   DeviceInfo
	available    bool
	useMPS       bool
	mpsThreshold int // Matrix size threshold for using MPS
}

// NewMetalBackend creates a new Metal backend instance
func NewMetalBackend(logger *slog.Logger) *MetalBackend {
	backend := &MetalBackend{
		logger:       logger,
		mpsThreshold: 512, // Use MPS for matrices larger than 512x512
	}
	
	// Check if Metal is available
	if err := backend.checkDevice(); err != nil {
		logger.Warn("Metal device not available", "error", err)
		backend.available = false
	} else {
		backend.available = true
	}
	
	return backend
}

// Initialize prepares the Metal backend for use
func (m *MetalBackend) Initialize() error {
	if !m.available {
		return fmt.Errorf("Metal device not available")
	}
	
	if m.initialized {
		return nil
	}
	
	m.logger.Debug("Initializing Metal backend")
	
	// Try to initialize with embedded library first
	embeddedLib := metal.GetMetalLib()
	var result C.int
	
	if len(embeddedLib) > 0 {
		m.logger.Debug("Attempting to load Metal library from embedded data", "size", len(embeddedLib))
		result = C.metal_init_with_embedded_lib(unsafe.Pointer(&embeddedLib[0]), C.size_t(len(embeddedLib)))
		if result == 0 {
			m.logger.Info("Successfully loaded Metal library from embedded data")
		} else {
			m.logger.Warn("Failed to load embedded Metal library, falling back to file loading", "error_code", result)
		}
	}
	
	// Fall back to file-based loading if embedded loading failed
	if result != 0 {
		m.logger.Debug("Loading Metal library from file")
		result = C.metal_init()
		if result != 0 {
			return fmt.Errorf("failed to initialize Metal: error code %d", result)
		}
	}
	
	// Get device information
	var info C.MetalDeviceInfo
	result = C.metal_get_device_info(&info)
	if result != 0 {
		return fmt.Errorf("failed to get device info: error code %d", result)
	}
	
	// Convert C device info to Go struct
	m.deviceInfo = DeviceInfo{
		Name:              C.GoString(&info.name[0]),
		TotalMemory:       int64(info.total_memory),
		AvailableMemory:   int64(info.available_memory),
		ComputeCapability: C.GoString(&info.gpu_family[0]),
		DriverVersion:     getMetalVersion(),
	}
	
	// Add Metal-specific info
	if info.is_unified_memory == 1 {
		m.deviceInfo.ComputeCapability += " (Unified Memory)"
	}
	
	// Check MPS availability
	m.useMPS = info.supports_mps == 1
	
	m.initialized = true
	m.logger.Info("Metal backend initialized with MPS support", 
		"device", m.deviceInfo.Name,
		"gpu_family", m.deviceInfo.ComputeCapability,
		"total_memory_gb", float64(m.deviceInfo.TotalMemory)/(1<<30),
		"available_memory_gb", float64(m.deviceInfo.AvailableMemory)/(1<<30),
		"unified_memory", info.is_unified_memory == 1,
		"mps_enabled", m.useMPS,
		"mps_threshold", m.mpsThreshold)
	
	return nil
}

// MatrixMultiply performs matrix multiplication using Metal
func (m *MetalBackend) MatrixMultiply(a, b []float32, mDim, k, n int) ([]float32, error) {
	if !m.initialized {
		if err := m.Initialize(); err != nil {
			return nil, fmt.Errorf("failed to initialize Metal backend: %w", err)
		}
	}
	
	// Validate dimensions
	if len(a) != mDim*k {
		return nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", mDim*k, len(a))
	}
	if len(b) != k*n {
		return nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", k*n, len(b))
	}
	
	// Allocate result matrix
	result := make([]float32, mDim*n)
	
	// Convert Go slices to C pointers
	var aPtr, bPtr, cPtr *C.float
	if len(a) > 0 {
		aPtr = (*C.float)(unsafe.Pointer(&a[0]))
	}
	if len(b) > 0 {
		bPtr = (*C.float)(unsafe.Pointer(&b[0]))
	}
	if len(result) > 0 {
		cPtr = (*C.float)(unsafe.Pointer(&result[0]))
	}
	
	// Decide whether to use MPS or custom kernel
	useMPS := m.useMPS && (mDim >= m.mpsThreshold || n >= m.mpsThreshold || k >= m.mpsThreshold)
	
	m.logger.Debug("Performing Metal matrix multiplication",
		"m", mDim, "k", k, "n", n,
		"flops", 2*mDim*k*n,
		"use_mps", useMPS)
	
	// Perform matrix multiplication
	var metalResult C.int
	if useMPS {
		// Use Metal Performance Shaders for large matrices
		metalResult = C.metal_matmul_mps(aPtr, bPtr, cPtr, C.int(mDim), C.int(n), C.int(k))
	} else {
		// Use custom kernel for small matrices
		// Use tiled kernel for medium-sized matrices
		useTiled := mDim >= 64 && n >= 64 && k >= 64
		metalResult = C.metal_matmul_kernel(aPtr, bPtr, cPtr, C.int(mDim), C.int(n), C.int(k), C.int(boolToInt(useTiled)))
	}
	
	if metalResult != 0 {
		return nil, fmt.Errorf("Metal matrix multiplication failed: error code %d", metalResult)
	}
	
	return result, nil
}

// GetDeviceInfo returns information about the Metal device
func (m *MetalBackend) GetDeviceInfo() DeviceInfo {
	return m.deviceInfo
}

// IsAvailable checks if Metal is available
func (m *MetalBackend) IsAvailable() bool {
	return m.available
}

// Cleanup releases Metal resources
func (m *MetalBackend) Cleanup() error {
	if !m.initialized {
		return nil
	}
	
	m.logger.Debug("Cleaning up Metal backend")
	
	result := C.metal_cleanup()
	if result != 0 {
		return fmt.Errorf("failed to cleanup Metal: error code %d", result)
	}
	
	m.initialized = false
	return nil
}

// checkDevice verifies Metal device availability
func (m *MetalBackend) checkDevice() error {
	result := C.metal_check_device()
	if result != 0 {
		return fmt.Errorf("Metal device check failed: no Metal-capable device found")
	}
	return nil
}

// SetMPSThreshold sets the matrix size threshold for using MPS
func (m *MetalBackend) SetMPSThreshold(threshold int) {
	m.mpsThreshold = threshold
	m.logger.Info("Updated MPS threshold", "threshold", threshold)
}

// getMetalVersion gets the Metal/macOS version
func getMetalVersion() string {
	// Get macOS version as a proxy for Metal version
	return runtime.GOOS + " " + runtime.GOARCH
}

// boolToInt converts bool to int for C interop
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}