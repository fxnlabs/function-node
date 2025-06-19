package gpu

// DeviceInfo contains information about the GPU device
type DeviceInfo struct {
	Name            string  `json:"name"`
	TotalMemory     int64   `json:"totalMemory"`     // in bytes
	AvailableMemory int64   `json:"availableMemory"` // in bytes
	ComputeCapability string `json:"computeCapability"`
	DriverVersion   string  `json:"driverVersion"`
	CUDAVersion     string  `json:"cudaVersion,omitempty"`
}

// GPUBackend defines the interface for GPU compute backends
// This interface allows for multiple GPU implementations (CUDA, ROCm, Metal, etc.)
// and provides a consistent API for matrix operations used in AI workload verification.
//
// Implementation notes:
// - Backends should handle memory management internally
// - Automatic fallback to CPU should be handled by the Manager, not the backend
// - Backends should be thread-safe for concurrent operations
// - Resource cleanup is critical to prevent GPU memory leaks
type GPUBackend interface {
	// MatrixMultiply performs matrix multiplication C = A * B
	// where A is m×k, B is k×n, and C is m×n
	//
	// Parameters:
	//   - a: First matrix in row-major order (size: m*k)
	//   - b: Second matrix in row-major order (size: k*n)
	//   - m: Number of rows in matrix A
	//   - k: Number of columns in A / rows in B
	//   - n: Number of columns in matrix B
	//
	// Returns:
	//   - Result matrix C in row-major order (size: m*n)
	//   - Error if the operation fails
	//
	// Performance notes:
	// - Small matrices (< 64x64) may be computed on CPU for efficiency
	// - Large matrices benefit from GPU acceleration
	// - Memory transfer overhead should be considered for backend selection
	MatrixMultiply(a, b []float32, m, k, n int) ([]float32, error)

	// GetDeviceInfo returns information about the GPU device
	// This information is used for:
	// - Reporting capabilities to schedulers
	// - Monitoring and metrics
	// - Debugging and troubleshooting
	GetDeviceInfo() DeviceInfo

	// IsAvailable checks if the backend is available for use
	// This should perform a quick check without heavy initialization
	// Used by the Manager to select appropriate backends
	IsAvailable() bool

	// Initialize prepares the backend for use
	// This may include:
	// - Creating CUDA contexts
	// - Allocating memory pools
	// - Warming up kernels
	// Should be called once before first use
	Initialize() error

	// Cleanup releases any resources held by the backend
	// Critical for preventing resource leaks:
	// - GPU memory allocations
	// - CUDA contexts
	// - Temporary buffers
	// Must be called when the backend is no longer needed
	Cleanup() error
}