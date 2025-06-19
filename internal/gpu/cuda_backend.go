// +build cuda

package gpu

/*
#cgo CFLAGS: -I../../cuda
#cgo LDFLAGS: -L../../cuda -lmatmul_cuda -lcudart -lcublas
#include "matmul.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"log/slog"
	"runtime"
	"unsafe"
)

// CUDABackend implements GPUBackend using NVIDIA CUDA
type CUDABackend struct {
	logger       *slog.Logger
	initialized  bool
	deviceInfo   DeviceInfo
	available    bool
}

// NewCUDABackend creates a new CUDA backend instance
func NewCUDABackend(logger *slog.Logger) *CUDABackend {
	backend := &CUDABackend{
		logger: logger,
	}
	
	// Check if CUDA is available
	if err := backend.checkDevice(); err != nil {
		logger.Warn("CUDA device not available", "error", err)
		backend.available = false
	} else {
		backend.available = true
	}
	
	return backend
}

// Initialize prepares the CUDA backend for use
func (c *CUDABackend) Initialize() error {
	if !c.available {
		return fmt.Errorf("CUDA device not available")
	}
	
	if c.initialized {
		return nil
	}
	
	c.logger.Debug("Initializing CUDA backend")
	
	// Initialize CUDA context
	result := C.cuda_init()
	if result != C.cudaSuccess {
		return fmt.Errorf("failed to initialize CUDA: %v", cudaErrorString(result))
	}
	
	// Get device information
	var info C.CudaDeviceInfo
	result = C.cuda_get_device_info(&info)
	if result != C.cudaSuccess {
		return fmt.Errorf("failed to get device info: %v", cudaErrorString(result))
	}
	
	// Convert C device info to Go struct
	c.deviceInfo = DeviceInfo{
		Name:              C.GoString(&info.name[0]),
		TotalMemory:       int64(info.total_memory),
		AvailableMemory:   int64(info.total_memory), // CUDA doesn't provide available memory easily
		ComputeCapability: fmt.Sprintf("%d.%d", int(info.major), int(info.minor)),
		DriverVersion:     getDriverVersion(),
		CUDAVersion:       getCUDAVersion(),
	}
	
	c.initialized = true
	c.logger.Info("CUDA backend initialized", 
		"device", c.deviceInfo.Name,
		"compute_capability", c.deviceInfo.ComputeCapability,
		"total_memory_gb", float64(c.deviceInfo.TotalMemory)/(1<<30))
	
	return nil
}

// MatrixMultiply performs matrix multiplication using CUDA
func (c *CUDABackend) MatrixMultiply(a, b []float32, m, k, n int) ([]float32, error) {
	if !c.initialized {
		if err := c.Initialize(); err != nil {
			return nil, fmt.Errorf("failed to initialize CUDA backend: %w", err)
		}
	}
	
	// Validate dimensions
	if len(a) != m*k {
		return nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", m*k, len(a))
	}
	if len(b) != k*n {
		return nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", k*n, len(b))
	}
	
	// Allocate result matrix
	result := make([]float32, m*n)
	
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
	
	c.logger.Debug("Performing CUDA matrix multiplication",
		"m", m, "k", k, "n", n,
		"flops", 2*m*k*n)
	
	// Perform matrix multiplication
	cudaResult := C.matmul_cuda(aPtr, bPtr, cPtr, C.int(m), C.int(n), C.int(k))
	if cudaResult != C.cudaSuccess {
		return nil, fmt.Errorf("CUDA matrix multiplication failed: %v", cudaErrorString(cudaResult))
	}
	
	return result, nil
}

// GetDeviceInfo returns information about the CUDA device
func (c *CUDABackend) GetDeviceInfo() DeviceInfo {
	return c.deviceInfo
}

// IsAvailable checks if CUDA is available
func (c *CUDABackend) IsAvailable() bool {
	return c.available
}

// Cleanup releases CUDA resources
func (c *CUDABackend) Cleanup() error {
	if !c.initialized {
		return nil
	}
	
	c.logger.Debug("Cleaning up CUDA backend")
	
	result := C.cuda_cleanup()
	if result != C.cudaSuccess {
		return fmt.Errorf("failed to cleanup CUDA: %v", cudaErrorString(result))
	}
	
	c.initialized = false
	return nil
}

// checkDevice verifies CUDA device availability
func (c *CUDABackend) checkDevice() error {
	result := C.cuda_check_device()
	if result != C.cudaSuccess {
		return fmt.Errorf("CUDA device check failed: %v", cudaErrorString(result))
	}
	return nil
}

// cudaErrorString converts CUDA error code to string
func cudaErrorString(err C.cudaError_t) string {
	switch err {
	case C.cudaSuccess:
		return "Success"
	case C.cudaErrorInvalidValue:
		return "Invalid value"
	case C.cudaErrorMemoryAllocation:
		return "Memory allocation failed"
	case C.cudaErrorInitializationError:
		return "Initialization error"
	case C.cudaErrorInsufficientDriver:
		return "Insufficient driver"
	case C.cudaErrorNoDevice:
		return "No CUDA device"
	default:
		return fmt.Sprintf("Unknown error (%d)", int(err))
	}
}

// getDriverVersion gets NVIDIA driver version
func getDriverVersion() string {
	// This would normally use nvidia-ml or similar
	// For now, return a placeholder
	return "Unknown"
}

// getCUDAVersion gets CUDA runtime version
func getCUDAVersion() string {
	// This would normally query the CUDA runtime
	// For now, return a placeholder
	return runtime.Version()
}