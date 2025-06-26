// +build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/fxnlabs/function-node/internal/auth"
	"github.com/fxnlabs/function-node/internal/challenge"
	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/gpu"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/fxnlabs/function-node/internal/metrics"
	"github.com/fxnlabs/function-node/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/fx"
	"go.uber.org/fx/fxtest"
)

// MockSchedulerRegistry for testing
type MockSchedulerRegistry struct{}

func (m *MockSchedulerRegistry) IsScheduler(address string) (bool, error) {
	// Accept test scheduler address
	return address == "0x1234567890123456789012345678901234567890", nil
}

func (m *MockSchedulerRegistry) Start() error {
	return nil
}

func (m *MockSchedulerRegistry) Stop() error {
	return nil
}

func TestMatrixChallenge_EndToEnd(t *testing.T) {
	// Setup test application
	var handler *challenge.ChallengeHandler
	var authMiddleware *auth.Middleware
	var testServer *httptest.Server
	
	app := fxtest.New(t,
		fx.Provide(
			func() *config.Config {
				return &config.Config{
					Listen: ":0",
					Logger: config.LoggerConfig{
						Level: "debug",
					},
				}
			},
			logger.NewLogger,
			metrics.NewMetrics,
			func() registry.SchedulerRegistry {
				return &MockSchedulerRegistry{}
			},
			func() (*keys.Keys, error) {
				// Use test scheduler key
				return keys.LoadKeysFromFile("../../scripts/scheduler_test_key.json")
			},
			auth.NewMiddleware,
			challenge.NewChallengeHandler,
		),
		fx.Populate(&handler, &authMiddleware),
	)
	
	app.RequireStart()
	defer app.RequireStop()
	
	// Create test server
	mux := http.NewServeMux()
	mux.HandleFunc("/challenge", authMiddleware.SchedulerAuth(handler.HandleChallenge))
	testServer = httptest.NewServer(mux)
	defer testServer.Close()
	
	// Test cases for different matrix sizes
	testCases := []struct {
		name         string
		payload      map[string]interface{}
		validateResp func(*testing.T, map[string]interface{})
	}{
		{
			name: "small fixed matrices",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"A": [][]float64{{1, 2}, {3, 4}},
					"B": [][]float64{{5, 6}, {7, 8}},
				},
			},
			validateResp: func(t *testing.T, resp map[string]interface{}) {
				assert.Equal(t, "success", resp["status"])
				
				result := resp["result"].(map[string]interface{})
				assert.Contains(t, result, "C")
				assert.Contains(t, result, "computationTimeMs")
				assert.Contains(t, result, "backend")
				
				// Verify result
				C := result["C"].([]interface{})
				assert.Equal(t, 2, len(C))
				row0 := C[0].([]interface{})
				assert.InDelta(t, 19.0, row0[0].(float64), 1e-5)
				assert.InDelta(t, 22.0, row0[1].(float64), 1e-5)
			},
		},
		{
			name: "random matrix generation",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"size": 50,
				},
			},
			validateResp: func(t *testing.T, resp map[string]interface{}) {
				assert.Equal(t, "success", resp["status"])
				
				result := resp["result"].(map[string]interface{})
				assert.Contains(t, result, "C")
				assert.Contains(t, result, "computationTimeMs")
				assert.Equal(t, float64(50), result["matrixSize"])
				assert.Equal(t, float64(250000), result["flops"]) // 2 * 50^3
			},
		},
		{
			name: "large matrix performance",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"size": 500,
				},
			},
			validateResp: func(t *testing.T, resp map[string]interface{}) {
				assert.Equal(t, "success", resp["status"])
				
				result := resp["result"].(map[string]interface{})
				assert.NotContains(t, result, "C") // Should not include large result
				assert.Contains(t, result, "computationTimeMs")
				
				// Check performance
				compTime := result["computationTimeMs"].(float64)
				flops := result["flops"].(float64)
				gflops := flops / (compTime / 1000.0) / 1e9
				
				t.Logf("500x500 matrix: %.2fms, %.2f GFLOPS", compTime, gflops)
			},
		},
		{
			name: "explicit backend selection",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"size":    100,
					"backend": "cpu",
				},
			},
			validateResp: func(t *testing.T, resp map[string]interface{}) {
				assert.Equal(t, "success", resp["status"])
				
				result := resp["result"].(map[string]interface{})
				backend := result["backend"].(string)
				assert.Equal(t, "cpu", backend)
			},
		},
	}
	
	// Load test keys
	testKeys, err := keys.LoadKeysFromFile("../../scripts/scheduler_test_key.json")
	require.NoError(t, err)
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create request
			body, err := json.Marshal(tc.payload)
			require.NoError(t, err)
			
			req, err := http.NewRequest("POST", testServer.URL+"/challenge", bytes.NewReader(body))
			require.NoError(t, err)
			
			// Sign request
			signature, err := keys.SignMessage(testKeys.PrivateKey, body)
			require.NoError(t, err)
			
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Signature", signature)
			req.Header.Set("X-Address", testKeys.Address)
			req.Header.Set("X-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
			
			// Send request
			client := &http.Client{Timeout: 30 * time.Second}
			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			
			// Read response
			respBody, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			
			assert.Equal(t, http.StatusOK, resp.StatusCode, "Response: %s", string(respBody))
			
			// Parse response
			var result map[string]interface{}
			err = json.Unmarshal(respBody, &result)
			require.NoError(t, err)
			
			// Validate response
			tc.validateResp(t, result)
		})
	}
}

func TestMatrixChallenge_ErrorHandling(t *testing.T) {
	// Test error cases
	var handler *challenge.ChallengeHandler
	var authMiddleware *auth.Middleware
	
	app := fxtest.New(t,
		fx.Provide(
			func() *config.Config {
				return &config.Config{
					Listen: ":0",
					Logger: config.LoggerConfig{
						Level: "debug",
					},
				}
			},
			logger.NewLogger,
			metrics.NewMetrics,
			func() registry.SchedulerRegistry {
				return &MockSchedulerRegistry{}
			},
			func() (*keys.Keys, error) {
				return keys.LoadKeysFromFile("../../scripts/scheduler_test_key.json")
			},
			auth.NewMiddleware,
			challenge.NewChallengeHandler,
		),
		fx.Populate(&handler, &authMiddleware),
	)
	
	app.RequireStart()
	defer app.RequireStop()
	
	// Create test server
	mux := http.NewServeMux()
	mux.HandleFunc("/challenge", authMiddleware.SchedulerAuth(handler.HandleChallenge))
	testServer := httptest.NewServer(mux)
	defer testServer.Close()
	
	testCases := []struct {
		name          string
		payload       map[string]interface{}
		expectedError string
	}{
		{
			name: "incompatible dimensions",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"A": [][]float64{{1, 2}},
					"B": [][]float64{{3, 4, 5}},
				},
			},
			expectedError: "not compatible for multiplication",
		},
		{
			name: "missing parameters",
			payload: map[string]interface{}{
				"type":    "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{},
			},
			expectedError: "either 'size' or both 'A' and 'B' matrices must be provided",
		},
		{
			name: "invalid matrix data",
			payload: map[string]interface{}{
				"type": "MATRIX_MULTIPLICATION",
				"payload": map[string]interface{}{
					"A": "not a matrix",
					"B": [][]float64{{1, 2}},
				},
			},
			expectedError: "json: cannot unmarshal",
		},
	}
	
	// Load test keys
	testKeys, err := keys.LoadKeysFromFile("../../scripts/scheduler_test_key.json")
	require.NoError(t, err)
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create request
			body, err := json.Marshal(tc.payload)
			require.NoError(t, err)
			
			req, err := http.NewRequest("POST", testServer.URL+"/challenge", bytes.NewReader(body))
			require.NoError(t, err)
			
			// Sign request
			signature, err := keys.SignMessage(testKeys.PrivateKey, body)
			require.NoError(t, err)
			
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Signature", signature)
			req.Header.Set("X-Address", testKeys.Address)
			req.Header.Set("X-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
			
			// Send request
			client := &http.Client{Timeout: 10 * time.Second}
			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			
			// Read response
			respBody, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			
			// Parse response
			var result map[string]interface{}
			err = json.Unmarshal(respBody, &result)
			require.NoError(t, err)
			
			assert.Equal(t, "error", result["status"])
			assert.Contains(t, result["error"].(string), tc.expectedError)
		})
	}
}

func BenchmarkMatrixChallenge(b *testing.B) {
	// Benchmark different matrix sizes
	sizes := []int{50, 100, 200, 500, 1000}
	
	logger := logger.NewLogger(&config.Config{
		Logger: config.LoggerConfig{Level: "error"},
	})
	
	challenger := challengers.NewMatrixMultiplicationChallenger()
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			payload := map[string]interface{}{
				"size": size,
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := challenger.Execute(payload, logger)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			// Calculate theoretical FLOPS
			flops := int64(2 * size * size * size * b.N)
			seconds := b.Elapsed().Seconds()
			gflops := float64(flops) / seconds / 1e9
			
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

func TestMatrixChallenge_GPUvsCPU(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping GPU vs CPU comparison in short mode")
	}
	
	logger := logger.NewLogger(&config.Config{
		Logger: config.LoggerConfig{Level: "debug"},
	})
	
	// Test GPU backend if available
	gpuBackend := gpu.NewGPUBackend(logger)
	defer gpuBackend.Cleanup()
	
	sizes := []int{256, 512, 1024}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test matrices
			a := make([]float32, size*size)
			b := make([]float32, size*size)
			
			for i := range a {
				a[i] = float32(i%100) / 100.0
				b[i] = float32((i+1)%100) / 100.0
			}
			
			// Test with GPU if available
			if gpuBackend.IsAvailable() {
				start := time.Now()
				gpuResult, err := gpuBackend.MatrixMultiply(a, b, size, size, size)
				gpuTime := time.Since(start)
				
				require.NoError(t, err)
				assert.Equal(t, size*size, len(gpuResult))
				
				flops := float64(2 * size * size * size)
				gpuGflops := flops / gpuTime.Seconds() / 1e9
				
				t.Logf("GPU: size=%d, time=%v, GFLOPS=%.2f", size, gpuTime, gpuGflops)
			}
			
			// Test with CPU backend
			cpuBackend := gpu.NewCPUBackend(logger)
			defer cpuBackend.Cleanup()
			
			start := time.Now()
			cpuResult, err := cpuBackend.MatrixMultiply(a, b, size, size, size)
			cpuTime := time.Since(start)
			
			require.NoError(t, err)
			assert.Equal(t, size*size, len(cpuResult))
			
			flops := float64(2 * size * size * size)
			cpuGflops := flops / cpuTime.Seconds() / 1e9
			
			t.Logf("CPU: size=%d, time=%v, GFLOPS=%.2f", size, cpuTime, cpuGflops)
			
			// If GPU is available, verify results match
			if gpuBackend.IsAvailable() {
				gpuResult, _ := gpuBackend.MatrixMultiply(a, b, size, size, size)
				
				// Compare a sample of results (not all for performance)
				for i := 0; i < 100 && i < len(cpuResult); i++ {
					assert.InDelta(t, cpuResult[i], gpuResult[i], 1e-4,
						"Results differ at index %d: CPU=%f, GPU=%f", i, cpuResult[i], gpuResult[i])
				}
			}
		})
	}
}