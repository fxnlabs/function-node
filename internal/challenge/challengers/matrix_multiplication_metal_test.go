//go:build metal && darwin
// +build metal,darwin

package challengers

import (
	"testing"

	"go.uber.org/zap"
)

func TestMatrixMultiplicationChallenger_WithMetal(t *testing.T) {
	log := zap.NewNop()
	challenger := NewMatrixMultiplicationChallenger()

	tests := []struct {
		name    string
		payload map[string]interface{}
		wantErr bool
	}{
		{
			name: "Small matrix with Metal",
			payload: map[string]interface{}{
				"size":    64,
				"backend": "gpu",
			},
			wantErr: false,
		},
		{
			name: "Large matrix with auto backend selection",
			payload: map[string]interface{}{
				"size": 512,
			},
			wantErr: false,
		},
		{
			name: "Force Metal backend",
			payload: map[string]interface{}{
				"size":    256,
				"backend": "gpu",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := challenger.Execute(tt.payload, log)
			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}

			if err == nil {
				// Check result structure
				resultMap, ok := result.(map[string]interface{})
				if !ok {
					t.Fatal("Result is not a map")
				}

				// Verify backend used
				backend, ok := resultMap["backend"].(string)
				if !ok {
					t.Fatal("Backend not found in result")
				}

				// Should use Metal backend for GPU requests on macOS
				if tt.payload["backend"] == "gpu" && backend != "metal" {
					t.Errorf("Expected Metal backend, got %s", backend)
				}

				// Check performance metrics
				computationTime, ok := resultMap["computationTimeMs"].(float64)
				if !ok || computationTime <= 0 {
					t.Error("Invalid computation time")
				}

				gflops, ok := resultMap["gflops"].(float64)
				if !ok || gflops <= 0 {
					t.Error("Invalid GFLOPS")
				}

				t.Logf("Backend: %s, Time: %.2fms, Performance: %.2f GFLOPS", 
					backend, computationTime, gflops)

				// Check device info
				deviceInfo, ok := resultMap["deviceInfo"].(map[string]interface{})
				if !ok {
					t.Fatal("Device info not found")
				}

				deviceName, ok := deviceInfo["name"].(string)
				if !ok || deviceName == "" {
					t.Error("Invalid device name")
				}

				t.Logf("Device: %s", deviceName)
			}
		})
	}
}

func TestMatrixMultiplicationChallenger_MetalPerformance(t *testing.T) {
	log := zap.NewNop()
	challenger := NewMatrixMultiplicationChallenger()

	// Test performance with increasing matrix sizes
	sizes := []int{128, 256, 512, 1024}
	
	for _, size := range sizes {
		t.Run(zap.Int("size", size).String, func(t *testing.T) {
			payload := map[string]interface{}{
				"size":    size,
				"backend": "gpu",
			}

			result, err := challenger.Execute(payload, log)
			if err != nil {
				t.Fatalf("Execute() failed: %v", err)
			}

			resultMap := result.(map[string]interface{})
			gflops := resultMap["gflops"].(float64)
			computationTime := resultMap["computationTimeMs"].(float64)
			backend := resultMap["backend"].(string)

			t.Logf("Size: %dx%d, Backend: %s, Time: %.2fms, Performance: %.2f GFLOPS",
				size, size, backend, computationTime, gflops)

			// Metal should provide significant speedup for large matrices
			if size >= 256 && backend == "metal" && gflops < 10.0 {
				t.Errorf("Performance too low for Metal backend: %.2f GFLOPS", gflops)
			}
		})
	}
}