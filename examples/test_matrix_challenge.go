package main

import (
	"crypto/ecdsa"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/gpu"
	"go.uber.org/zap"
)

type BenchmarkResult struct {
	Size     int
	Backend  string
	Duration time.Duration
	GFLOPS   float64
}

func generateMatrices(size int) ([]float32, []float32) {
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	
	for i := 0; i < size*size; i++ {
		a[i] = float32(i%100) * 0.01
		b[i] = float32((i*2+1)%100) * 0.01
	}
	
	return a, b
}

func calculateGFLOPS(size int, duration time.Duration) float64 {
	// Matrix multiplication requires 2*M*N*K floating point operations
	// For square matrices: 2*size^3
	ops := float64(2 * size * size * size)
	seconds := duration.Seconds()
	return ops / (seconds * 1e9) // Convert to GFLOPS
}

func runBenchmark(backend gpu.GPUBackend, size int, iterations int) (time.Duration, error) {
	a, b := generateMatrices(size)
	
	var totalDuration time.Duration
	
	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := backend.MatrixMultiply(a, b, size, size, size)
		if err != nil {
			return 0, err
		}
		totalDuration += time.Since(start)
	}
	
	return totalDuration / time.Duration(iterations), nil
}

func printBenchmarkTable(results []BenchmarkResult) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("                        MATRIX MULTIPLICATION BENCHMARK")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("│ %-8s │ %-12s │ %-12s │ %-12s │ %-12s │\n", 
		"Size", "Backend", "Time (ms)", "GFLOPS", "Speedup")
	fmt.Println("├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
	
	// Group results by size for speedup calculation
	sizeGroups := make(map[int][]BenchmarkResult)
	for _, result := range results {
		sizeGroups[result.Size] = append(sizeGroups[result.Size], result)
	}
	
	for _, size := range []int{256, 512, 1024} {
		if group, exists := sizeGroups[size]; exists {
			var cpuTime time.Duration
			
			// Find CPU baseline time
			for _, result := range group {
				if result.Backend == "CPU" {
					cpuTime = result.Duration
					break
				}
			}
			
			for i, result := range group {
				var speedup string
				if result.Backend == "CPU" {
					speedup = "1.00x"
				} else if cpuTime > 0 {
					speedup = fmt.Sprintf("%.2fx", float64(cpuTime)/float64(result.Duration))
				} else {
					speedup = "N/A"
				}
				
				if i == 0 {
					fmt.Printf("│ %-8d │ %-12s │ %9.2f ms │ %9.2f    │ %-12s │\n",
						result.Size, result.Backend, 
						float64(result.Duration.Nanoseconds())/1e6,
						result.GFLOPS, speedup)
				} else {
					fmt.Printf("│ %-8s │ %-12s │ %9.2f ms │ %9.2f    │ %-12s │\n",
						"", result.Backend, 
						float64(result.Duration.Nanoseconds())/1e6,
						result.GFLOPS, speedup)
				}
			}
			if size != 1024 {
				fmt.Println("├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
			}
		}
	}
	fmt.Println("└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
	fmt.Println()
}

func main() {
	// Create logger (silent for cleaner output)
	logger, _ := zap.NewDevelopment()
	defer logger.Sync()

	// Load or generate private key
	var privateKey *ecdsa.PrivateKey
	if keyPath := os.Getenv("PRIVATE_KEY_PATH"); keyPath != "" {
		keyBytes, err := os.ReadFile(keyPath)
		if err == nil {
			privateKey, _ = crypto.HexToECDSA(string(keyBytes))
		}
	}
	if privateKey == nil {
		privateKey, _ = crypto.GenerateKey()
	}

	fmt.Println("🚀 Function Node Matrix Multiplication Challenge Demo")
	fmt.Println("═══════════════════════════════════════════════════════")

	// Test IDENTITY challenge first to see GPU info
	fmt.Println("\n📋 System Information")
	fmt.Println("─────────────────────")
	identityChallenger, _ := challenge.NewChallenger("IDENTITY", privateKey)
	identityResp, err := identityChallenger.Execute(struct{}{}, logger)
	if err != nil {
		fmt.Printf("❌ Error getting system info: %v\n", err)
	} else {
		if respMap, ok := identityResp.(map[string]interface{}); ok {
			if gpu_info, exists := respMap["gpu_info"]; exists {
				if gpuMap, ok := gpu_info.(map[string]interface{}); ok {
					if available, ok := gpuMap["available"].(bool); ok && available {
						fmt.Printf("✅ CUDA GPU Available: %s\n", gpuMap["device_name"])
						fmt.Printf("   Compute Capability: %s\n", gpuMap["compute_capability"])
						fmt.Printf("   Total Memory: %s\n", gpuMap["total_memory"])
					} else {
						fmt.Printf("⚠️  GPU acceleration not available - using CPU fallback\n")
					}
				}
			}
		}
	}

	// Initialize GPU backends
	fmt.Println("\n🔧 Initializing Backends")
	fmt.Println("────────────────────────")
	
	// Create a silent logger for cleaner output
	silentLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	gpuManager, err := gpu.NewManager(silentLogger)
	if err != nil {
		fmt.Printf("⚠️  Failed to initialize GPU manager: %v\n", err)
		return
	}
	
	// Test with different matrix sizes
	sizes := []int{256, 512, 1024}
	iterations := 3
	
	var results []BenchmarkResult
	
	fmt.Println("\n⏱️  Running Benchmarks")
	fmt.Println("─────────────────────")
	
	for _, size := range sizes {
		fmt.Printf("Testing %dx%d matrices...\n", size, size)
		
		// Test CPU backend
		cpuBackend := gpu.NewCPUBackend(silentLogger)
		cpuBackend.Initialize()
		fmt.Printf("  Running CPU benchmark... ")
		cpuDuration, err := runBenchmark(cpuBackend, size, iterations)
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
		} else {
			fmt.Printf("✅ %.2f ms\n", float64(cpuDuration.Nanoseconds())/1e6)
			results = append(results, BenchmarkResult{
				Size:     size,
				Backend:  "CPU",
				Duration: cpuDuration,
				GFLOPS:   calculateGFLOPS(size, cpuDuration),
			})
		}
		
		// Test current backend (could be CUDA or CPU fallback)
		currentBackend := gpuManager.GetBackend()
		if currentBackend != cpuBackend {
			backendName := "CUDA"
			fmt.Printf("  Running %s benchmark... ", backendName)
			gpuDuration, err := runBenchmark(currentBackend, size, iterations)
			if err != nil {
				fmt.Printf("❌ Error: %v\n", err)
			} else {
				fmt.Printf("✅ %.2f ms\n", float64(gpuDuration.Nanoseconds())/1e6)
				results = append(results, BenchmarkResult{
					Size:     size,
					Backend:  backendName,
					Duration: gpuDuration,
					GFLOPS:   calculateGFLOPS(size, gpuDuration),
				})
			}
		}
	}
	
	// Print benchmark results table
	if len(results) > 0 {
		printBenchmarkTable(results)
	}

	// Demonstrate the challenge system with a small example
	fmt.Println("🎯 Challenge System Demo")
	fmt.Println("────────────────────────")
	
	matrixChallenger, _ := challenge.NewChallenger("MATRIX_MULTIPLICATION", privateKey)
	
	// Test with specific small matrices to verify correctness
	fmt.Println("\nTesting 2x2 matrix multiplication:")
	fmt.Println("A = [[1, 2], [3, 4]]")
	fmt.Println("B = [[5, 6], [7, 8]]")
	fmt.Println("Expected: C = [[19, 22], [43, 50]]")
	
	payloadSpecific := map[string]interface{}{
		"size": 2,
		"matrixA": [][]float64{{1, 2}, {3, 4}},
		"matrixB": [][]float64{{5, 6}, {7, 8}},
	}
	
	start := time.Now()
	respSpecific, err := matrixChallenger.Execute(payloadSpecific, logger)
	duration := time.Since(start)
	
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
	} else {
		fmt.Printf("✅ Challenge completed in %v\n", duration)
		if respMap, ok := respSpecific.(map[string]interface{}); ok {
			if result, exists := respMap["result"]; exists {
				fmt.Printf("Result: %v\n", result)
			}
			if verified, exists := respMap["verified"]; exists {
				if verified.(bool) {
					fmt.Println("🔍 Result verification: ✅ PASSED")
				} else {
					fmt.Println("🔍 Result verification: ❌ FAILED")
				}
			}
		}
	}
	
	fmt.Println("\n🏁 Demo completed!")
}