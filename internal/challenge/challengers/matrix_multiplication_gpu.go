package challengers

import (
	"fmt"
	"log/slog"
	"time"

	"github.com/fxnlabs/function-node/internal/gpu"
	"go.uber.org/zap"
)

// multiplyMatricesGPU performs matrix multiplication using the GPU backend
func (c *MatrixMultiplicationChallenger) multiplyMatricesGPU(matrixA, matrixB [][]float64, manager *gpu.Manager, log *zap.Logger) ([][]float64, error) {
	aRows, aCols := len(matrixA), len(matrixA[0])
	bRows, bCols := len(matrixB), len(matrixB[0])

	if aCols != bRows {
		return nil, fmt.Errorf("matrix dimensions are not compatible for multiplication")
	}

	log.Debug("Starting GPU matrix multiplication",
		zap.Int("a_rows", aRows),
		zap.Int("a_cols", aCols),
		zap.Int("b_cols", bCols))

	// Time conversion separately
	conversionStart := time.Now()
	// Convert matrices to flat float32 arrays
	aFlat := gpu.Float64MatrixToFloat32(matrixA)
	bFlat := gpu.Float64MatrixToFloat32(matrixB)
	conversionTime := time.Since(conversionStart)
	
	log.Debug("Matrix conversion completed",
		zap.Duration("conversion_time", conversionTime),
		zap.Int("total_elements", aRows*aCols + bRows*bCols))

	// Time GPU computation
	computeStart := time.Now()
	
	// Perform GPU multiplication
	resultFlat, err := manager.MatrixMultiply(aFlat, bFlat, aRows, aCols, bCols)
	if err != nil {
		log.Error("GPU matrix multiplication failed", zap.Error(err))
		return nil, fmt.Errorf("GPU matrix multiplication failed: %w", err)
	}

	computeTime := time.Since(computeStart)
	
	// Log device info and timing
	deviceInfo := manager.GetDeviceInfo()
	log.Debug("GPU computation completed", 
		zap.Duration("compute_time", computeTime),
		zap.String("device", deviceInfo.Name),
		zap.Int64("memory_used_mb", (deviceInfo.TotalMemory-deviceInfo.AvailableMemory)/(1024*1024)))

	// Convert result back to 2D float64 matrix
	resultMatrix := gpu.Float32ArrayToFloat64Matrix(resultFlat, aRows, bCols)
	if resultMatrix == nil {
		log.Error("Failed to convert result matrix from GPU format")
		return nil, fmt.Errorf("failed to convert result matrix")
	}

	totalTime := time.Since(conversionStart)
	log.Info("GPU matrix multiplication completed successfully",
		zap.Duration("total_time", totalTime),
		zap.Duration("conversion_time", conversionTime),
		zap.Duration("compute_time", computeTime))

	return resultMatrix, nil
}

// initializeGPUManager creates and initializes a GPU manager
func initializeGPUManager(log *zap.Logger) (*gpu.Manager, error) {
	// Convert zap.Logger to slog.Logger
	// For now, we'll just use the default slog logger
	slogger := slog.Default()
	
	manager, err := gpu.NewManager(slogger)
	if err != nil {
		log.Error("Failed to create GPU manager", zap.Error(err))
		return nil, err
	}

	info := manager.GetDeviceInfo()
	log.Info("GPU backend initialized",
		zap.String("backend", manager.GetBackendType()),
		zap.String("device", info.Name),
		zap.String("compute_capability", info.ComputeCapability))

	if info.CUDAVersion != "" {
		log.Info("CUDA information",
			zap.String("cuda_version", info.CUDAVersion),
			zap.String("driver_version", info.DriverVersion),
			zap.Int64("total_memory_mb", info.TotalMemory/(1024*1024)),
			zap.Int64("available_memory_mb", info.AvailableMemory/(1024*1024)))
	}

	return manager, nil
}