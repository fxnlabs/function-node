package cuda

/*
#cgo CFLAGS: -I. -I./include
#cgo LDFLAGS: -L./lib -L. -lmatmul_cuda -lcudart

#include "matmul.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// DeviceInfo represents CUDA device information
type DeviceInfo struct {
	Name                   string
	Major                  int
	Minor                  int
	MemoryClockRate        int    // in KHz
	MemoryBusWidth         int    // in bits
	TotalMemory            uint64 // in bytes
	SharedMemoryPerBlock   uint64
	MaxThreadsPerBlock     int
	MaxThreadsDim          [3]int
	MaxGridSize            [3]int
	ClockRate              int // in KHz
	MultiProcessorCount    int
	ComputeCapability      int
}

// MatMul performs matrix multiplication C = A * B using CUDA
func MatMul(A, B []float32, M, N, K int) ([]float32, error) {
	if len(A) != M*K {
		return nil, fmt.Errorf("matrix A size mismatch: expected %d, got %d", M*K, len(A))
	}
	if len(B) != K*N {
		return nil, fmt.Errorf("matrix B size mismatch: expected %d, got %d", K*N, len(B))
	}

	C := make([]float32, M*N)

	// Convert slices to C pointers
	aPtr := (*C.float)(unsafe.Pointer(&A[0]))
	bPtr := (*C.float)(unsafe.Pointer(&B[0]))
	cPtr := (*C.float)(unsafe.Pointer(&C[0]))

	// Call CUDA function
	err := C.matmul_cuda(aPtr, bPtr, cPtr, C.int(M), C.int(N), C.int(K))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("CUDA matrix multiplication failed: %s", getCudaErrorString(err))
	}

	return C, nil
}

// GetDeviceInfo returns information about the current CUDA device
func GetDeviceInfo() (*DeviceInfo, error) {
	var cInfo C.CudaDeviceInfo
	
	err := C.cuda_get_device_info(&cInfo)
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("failed to get device info: %s", getCudaErrorString(err))
	}

	info := &DeviceInfo{
		Name:                 C.GoString(&cInfo.name[0]),
		Major:                int(cInfo.major),
		Minor:                int(cInfo.minor),
		MemoryClockRate:      int(cInfo.memory_clock_rate),
		MemoryBusWidth:       int(cInfo.memory_bus_width),
		TotalMemory:          uint64(cInfo.total_memory),
		SharedMemoryPerBlock: uint64(cInfo.shared_memory_per_block),
		MaxThreadsPerBlock:   int(cInfo.max_threads_per_block),
		ClockRate:            int(cInfo.clock_rate),
		MultiProcessorCount:  int(cInfo.multi_processor_count),
		ComputeCapability:    int(cInfo.compute_capability),
	}

	for i := 0; i < 3; i++ {
		info.MaxThreadsDim[i] = int(cInfo.max_threads_dim[i])
		info.MaxGridSize[i] = int(cInfo.max_grid_size[i])
	}

	return info, nil
}

// CheckDevice checks if a CUDA device is available
func CheckDevice() error {
	err := C.cuda_check_device()
	if err != C.cudaSuccess {
		return fmt.Errorf("CUDA device check failed: %s", getCudaErrorString(err))
	}
	return nil
}

// Init initializes the CUDA context
func Init() error {
	err := C.cuda_init()
	if err != C.cudaSuccess {
		return fmt.Errorf("CUDA initialization failed: %s", getCudaErrorString(err))
	}
	return nil
}

// Cleanup releases CUDA resources
func Cleanup() error {
	err := C.cuda_cleanup()
	if err != C.cudaSuccess {
		return fmt.Errorf("CUDA cleanup failed: %s", getCudaErrorString(err))
	}
	return nil
}

// getCudaErrorString converts CUDA error code to string
func getCudaErrorString(err C.cudaError_t) string {
	// Map common CUDA errors
	switch err {
	case 0: // cudaSuccess
		return "Success"
	case 1: // cudaErrorInvalidValue
		return "Invalid value"
	case 2: // cudaErrorMemoryAllocation
		return "Memory allocation failed"
	case 3: // cudaErrorInitializationError
		return "Initialization error"
	case 35: // cudaErrorInsufficientDriver
		return "Insufficient driver"
	case 100: // cudaErrorNoDevice
		return "No CUDA device"
	case 101: // cudaErrorInvalidDevice
		return "Invalid device"
	default:
		return fmt.Sprintf("CUDA error %d", int(err))
	}
}