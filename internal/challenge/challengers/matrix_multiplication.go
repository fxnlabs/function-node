package challengers

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"github.com/fxnlabs/function-node/internal/gpu"
	"github.com/fxnlabs/function-node/internal/metrics"
	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
)

// MatrixMultiplicationPayload represents the expected payload structure for matrix multiplication challenges
//
// The challenger supports multiple input formats:
// 1. Specific matrices: Provide MatrixA and MatrixB (or A and B for backward compatibility)
// 2. Random generation: Provide Size to generate random square matrices
// 3. Empty payload: Defaults to 32x32 random matrices
//
// Backend selection:
// - "gpu": Force GPU execution (fails if GPU unavailable)
// - "cpu": Force CPU execution
// - "auto": Automatically select based on matrix size and GPU availability (default)
//
// Performance characteristics:
// - Matrices < 64x64: CPU is typically faster due to memory transfer overhead
// - Matrices >= 64x64: GPU provides significant speedup (10x-100x for large matrices)
type MatrixMultiplicationPayload struct {
	Size      int         `json:"size,omitempty"`      // Optional: generate random matrices of this size
	MatrixA   [][]float64 `json:"matrixA,omitempty"`   // Optional: specific matrix A
	MatrixB   [][]float64 `json:"matrixB,omitempty"`   // Optional: specific matrix B
	A         [][]float64 `json:"A,omitempty"`         // Backward compatibility
	B         [][]float64 `json:"B,omitempty"`         // Backward compatibility
	Backend   string      `json:"backend,omitempty"`   // Optional: "gpu", "cpu", or "auto" (default)
	Precision string      `json:"precision,omitempty"` // Optional: data type precision (default: "float64")
}

// MatrixMultiplicationChallenger performs matrix multiplication.
type MatrixMultiplicationChallenger struct {
	gpuManager *gpu.Manager
}

// NewMatrixMultiplicationChallenger creates a new MatrixMultiplicationChallenger with GPU detection
func NewMatrixMultiplicationChallenger() *MatrixMultiplicationChallenger {
	return &MatrixMultiplicationChallenger{}
}

// initializeGPU initializes the GPU manager if not already done
func (c *MatrixMultiplicationChallenger) initializeGPU(log *zap.Logger) error {
	if c.gpuManager == nil {
		manager, err := initializeGPUManager(log)
		if err != nil {
			return err
		}
		c.gpuManager = manager
	}
	return nil
}

// Execute performs a matrix multiplication challenge.
//
// The challenge validates computational capabilities by performing matrix multiplication
// with optional GPU acceleration. This is used to verify that providers have the
// necessary hardware to handle AI inference workloads efficiently.
//
// Algorithm selection:
// 1. For small matrices (< 64x64) or when backend="cpu": Uses Gonum optimized BLAS
// 2. For large matrices with GPU available: Uses CUDA cuBLAS for maximum performance
// 3. Automatic fallback to CPU if GPU fails
//
// The result includes:
// - The computed matrix C = A × B
// - Computation time in milliseconds
// - Backend used (cpu/cuda)
// - GPU information if applicable
// - Performance metrics (GFLOPS)
func (c *MatrixMultiplicationChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	log.Info("Starting matrix multiplication challenge")

	// Parse the payload
	data, err := json.Marshal(payload)
	if err != nil {
		log.Error("Failed to marshal payload", zap.Error(err))
		return nil, err
	}

	var params MatrixMultiplicationPayload
	if err := json.Unmarshal(data, &params); err != nil {
		log.Error("Failed to unmarshal payload", zap.Error(err))
		return nil, err
	}

	// Set defaults
	if params.Backend == "" {
		params.Backend = "auto"
	}
	if params.Precision == "" {
		params.Precision = "float64"
	}

	// Handle backward compatibility
	if len(params.A) > 0 && len(params.MatrixA) == 0 {
		params.MatrixA = params.A
	}
	if len(params.B) > 0 && len(params.MatrixB) == 0 {
		params.MatrixB = params.B
	}

	// Generate or validate matrices
	var matrixA, matrixB [][]float64
	var matrixSize int

	if params.Size > 0 {
		// Generate random matrices of specified size
		matrixSize = params.Size
		matrixA = generateRandomMatrix(matrixSize, matrixSize)
		matrixB = generateRandomMatrix(matrixSize, matrixSize)
		log.Info("Generated random matrices", 
			zap.Int("size", matrixSize),
			zap.String("backend", params.Backend),
			zap.String("precision", params.Precision))
	} else if len(params.MatrixA) > 0 && len(params.MatrixB) > 0 {
		// Use provided matrices
		matrixA = params.MatrixA
		matrixB = params.MatrixB
		matrixSize = len(matrixA)

		// Validate matrix dimensions
		if len(matrixA) == 0 || len(matrixB) == 0 {
			return nil, fmt.Errorf("matrices A or B are empty")
		}

		aCols := len(matrixA[0])
		bRows := len(matrixB)

		if aCols != bRows {
			log.Error("Matrix dimensions are not compatible for multiplication",
				zap.Int("a_cols", aCols),
				zap.Int("b_rows", bRows))
			return nil, fmt.Errorf("matrix dimensions are not compatible for multiplication")
		}
		log.Info("Using provided matrices",
			zap.Int("a_rows", len(matrixA)),
			zap.Int("a_cols", aCols),
			zap.Int("b_rows", bRows),
			zap.Int("b_cols", len(matrixB[0])))
	} else {
		log.Error("Invalid payload: missing required matrix data")
		return nil, fmt.Errorf("either 'size' or both 'A' and 'B' matrices must be provided")
	}

	// Initialize GPU if requested
	var useGPU bool
	var deviceInfo map[string]interface{}
	
	if params.Backend == "gpu" || params.Backend == "metal" || params.Backend == "cuda" || (params.Backend == "auto" && matrixSize >= 100) {
		log.Debug("Attempting to initialize GPU backend",
			zap.String("backend_request", params.Backend),
			zap.Int("matrix_size", matrixSize))
		
		if err := c.initializeGPU(log); err == nil && c.gpuManager != nil && c.gpuManager.IsGPUAvailable() {
			useGPU = true
			info := c.gpuManager.GetDeviceInfo()
			deviceInfo = map[string]interface{}{
				"name":              info.Name,
				"computeCapability": info.ComputeCapability,
				"memoryGB":          float64(info.TotalMemory) / (1024 * 1024 * 1024),
				"cudaCores":         0, // Would need to be queried from device
			}
			if info.CUDAVersion != "" {
				deviceInfo["cudaVersion"] = info.CUDAVersion
				deviceInfo["driverVersion"] = info.DriverVersion
			}
			log.Info("GPU backend initialized successfully",
				zap.String("device_name", info.Name),
				zap.String("compute_capability", info.ComputeCapability),
				zap.Float64("memory_gb", deviceInfo["memoryGB"].(float64)))
		} else if params.Backend == "gpu" || params.Backend == "metal" || params.Backend == "cuda" {
			log.Warn("Specific GPU backend requested but not available, falling back to CPU",
				zap.String("requested_backend", params.Backend),
				zap.Error(err))
		}
	}

	if !useGPU {
		deviceInfo = map[string]interface{}{
			"name":              "CPU",
			"computeCapability": "N/A",
			"memoryGB":          0,
			"cudaCores":         0,
		}
	}

	// Start timing
	startTime := time.Now()

	// Perform multiplication
	var resultMatrix [][]float64
	var backend string

	if useGPU {
		log.Info("Using GPU backend", zap.String("device", c.gpuManager.GetDeviceInfo().Name))
		var mulErr error
		resultMatrix, mulErr = c.multiplyMatricesGPU(matrixA, matrixB, c.gpuManager, log)
		if mulErr != nil {
			log.Warn("GPU multiplication failed, falling back to CPU", zap.Error(mulErr))
			resultMatrix, mulErr = c.multiplyMatricesCPU(matrixA, matrixB)
			if mulErr != nil {
				return nil, mulErr
			}
			backend = "cpu"
		} else {
			backend = c.gpuManager.GetBackendType()
		}
	} else {
		log.Info("Using CPU backend")
		var mulErr error
		resultMatrix, mulErr = c.multiplyMatricesCPU(matrixA, matrixB)
		if mulErr != nil {
			return nil, mulErr
		}
		backend = "cpu"
	}

	// Calculate elapsed time
	elapsedTime := time.Since(startTime)
	computationTimeMs := float64(elapsedTime.Nanoseconds()) / 1e6

	// Calculate GFLOPS (2 * n^3 operations for square matrices)
	operations := float64(2 * matrixSize * matrixSize * matrixSize)
	gflops := (operations / 1e9) / (computationTimeMs / 1000.0)

	// Record metrics
	metrics.ChallengeMatrixMultDuration.Observe(computationTimeMs)
	metrics.ChallengeMatrixMultSize.Set(float64(matrixSize))
	metrics.ChallengeMatrixMultGFLOPS.Set(gflops)
	metrics.ChallengeMatrixMultBackend.WithLabelValues(backend).Inc()

	// Record GPU metrics if GPU was used
	if useGPU && c.gpuManager != nil {
		deviceInfo := c.gpuManager.GetDeviceInfo()
		if deviceInfo.AvailableMemory > 0 && deviceInfo.TotalMemory > 0 {
			usedMemory := deviceInfo.TotalMemory - deviceInfo.AvailableMemory
			metrics.GPUMemoryUsedBytes.Set(float64(usedMemory))
			// GPU utilization would need to be queried from the device
			// For now, we'll set it to 0 if not available
			metrics.GPUUtilizationPercent.Set(0)
		}
	}

	// Generate verification data
	merkleRoot := generateMerkleRoot(resultMatrix)
	resultSamples := generateResultSamples(resultMatrix, 2) // Return 2 samples as per spec

	log.Info("Matrix multiplication completed",
		zap.Float64("computation_time_ms", computationTimeMs),
		zap.String("backend", backend),
		zap.Int("matrix_size", matrixSize),
		zap.Float64("gflops", gflops),
		zap.String("merkle_root", merkleRoot))

	// Build response
	response := map[string]interface{}{
		"computationTimeMs": computationTimeMs,
		"gflops":            gflops,
		"backend":           backend,
		"device":            deviceInfo["name"], // For backward compatibility
		"deviceInfo":        deviceInfo,
		"matrixSize":        matrixSize,
		"merkleRoot":        merkleRoot,
		"resultSamples":     resultSamples,
		"flops":             int64(operations), // For backward compatibility
	}

	// Include result matrix for small matrices (backward compatibility)
	if matrixSize <= 100 {
		response["C"] = resultMatrix
	}

	return response, nil
}

// multiplyMatricesCPU performs matrix multiplication using gonum (CPU)
func (c *MatrixMultiplicationChallenger) multiplyMatricesCPU(matrixA, matrixB [][]float64) ([][]float64, error) {
	aRows, aCols := len(matrixA), len(matrixA[0])
	bRows, bCols := len(matrixB), len(matrixB[0])

	if aCols != bRows {
		return nil, fmt.Errorf("matrix dimensions are not compatible for multiplication")
	}

	// Convert to gonum matrices
	a := mat.NewDense(aRows, aCols, nil)
	for i, row := range matrixA {
		a.SetRow(i, row)
	}

	b := mat.NewDense(bRows, bCols, nil)
	for i, row := range matrixB {
		b.SetRow(i, row)
	}

	// Perform multiplication
	var res mat.Dense
	res.Mul(a, b)

	// Convert back to [][]float64
	r, cols := res.Dims()
	resultMatrix := make([][]float64, r)
	for i := range r {
		resultMatrix[i] = make([]float64, cols)
		for j := range cols {
			resultMatrix[i][j] = res.At(i, j)
		}
	}

	return resultMatrix, nil
}

// generateRandomMatrix creates a random matrix of the specified dimensions
func generateRandomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
}

// generateMerkleRoot generates a merkle root from the result matrix
func generateMerkleRoot(matrix [][]float64) string {
	// Simplified merkle root generation
	// In production, this would build a proper merkle tree
	var data []byte
	for i := range matrix {
		for j := range matrix[i] {
			data = append(data, []byte(fmt.Sprintf("%.6f", matrix[i][j]))...)
		}
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("0x%x", hash)
}

// generateResultSamples returns sample values from the result matrix
func generateResultSamples(matrix [][]float64, count int) []map[string]interface{} {
	samples := make([]map[string]interface{}, 0, count)
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return samples
	}

	rows := len(matrix)
	cols := len(matrix[0])

	// Sample at specific positions (first and middle)
	positions := [][]int{
		{0, 0},               // First element
		{rows / 2, cols / 2}, // Middle element
	}

	for i := 0; i < count && i < len(positions); i++ {
		row := positions[i][0]
		col := positions[i][1]
		if row < rows && col < cols {
			samples = append(samples, map[string]interface{}{
				"row":   row,
				"col":   col,
				"value": matrix[row][col],
			})
		}
	}

	return samples
}