package gpu

import (
	"log/slog"
	"math"
	"strings"
	"testing"
)

func TestCPUBackend(t *testing.T) {
	logger := slog.Default()
	backend := NewCPUBackend(logger)
	
	// Test initialization
	err := backend.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize CPU backend: %v", err)
	}
	defer backend.Cleanup()
	
	// Test availability
	if !backend.IsAvailable() {
		t.Error("CPU backend should always be available")
	}
	
	// Test device info
	info := backend.GetDeviceInfo()
	if !strings.Contains(info.Name, "CPU") {
		t.Errorf("Expected device name to contain 'CPU', got %s", info.Name)
	}
	
	// Test matrix multiplication
	// A: 2x3 matrix
	// B: 3x2 matrix
	// C: 2x2 matrix (result)
	a := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	b := []float32{
		7, 8,
		9, 10,
		11, 12,
	}
	
	result, err := backend.MatrixMultiply(a, b, 2, 3, 2)
	if err != nil {
		t.Fatalf("Matrix multiplication failed: %v", err)
	}
	
	// Expected result:
	// [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
	// [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
	expected := []float32{58, 64, 139, 154}
	
	if len(result) != len(expected) {
		t.Fatalf("Result length mismatch: expected %d, got %d", len(expected), len(result))
	}
	
	for i := range result {
		if math.Abs(float64(result[i]-expected[i])) > 1e-6 {
			t.Errorf("Result mismatch at index %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestManager(t *testing.T) {
	logger := slog.Default()
	manager, err := NewManager(logger)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Cleanup()
	
	// Test that we have a backend
	backend := manager.GetBackend()
	if backend == nil {
		t.Fatal("Manager should have a backend")
	}
	
	// Test device info
	info := manager.GetDeviceInfo()
	if info.Name == "" {
		t.Error("Device info should have a name")
	}
	
	// Test backend type
	backendType := manager.GetBackendType()
	if backendType != "cpu" && backendType != "cuda" && backendType != "metal" {
		t.Errorf("Unexpected backend type: %s", backendType)
	}
	
	// Test matrix multiplication through manager
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	
	result, err := manager.MatrixMultiply(a, b, 2, 2, 2)
	if err != nil {
		t.Fatalf("Matrix multiplication failed: %v", err)
	}
	
	// Expected: [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
	//          [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
	expected := []float32{19, 22, 43, 50}
	
	for i := range result {
		if math.Abs(float64(result[i]-expected[i])) > 1e-6 {
			t.Errorf("Result mismatch at index %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func TestUtilityFunctions(t *testing.T) {
	// Test Float64ToFloat32
	input64 := []float64{1.0, 2.0, 3.0, 4.0}
	output32 := Float64ToFloat32(input64)
	
	if len(output32) != len(input64) {
		t.Errorf("Length mismatch in Float64ToFloat32")
	}
	
	for i := range output32 {
		if float32(input64[i]) != output32[i] {
			t.Errorf("Value mismatch at index %d", i)
		}
	}
	
	// Test Float32ToFloat64
	input32 := []float32{1.0, 2.0, 3.0, 4.0}
	output64 := Float32ToFloat64(input32)
	
	if len(output64) != len(input32) {
		t.Errorf("Length mismatch in Float32ToFloat64")
	}
	
	for i := range output64 {
		if float64(input32[i]) != output64[i] {
			t.Errorf("Value mismatch at index %d", i)
		}
	}
	
	// Test matrix conversions
	matrix := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	
	flatArray := Float64MatrixToFloat32(matrix)
	expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	
	if len(flatArray) != len(expected) {
		t.Errorf("Length mismatch in Float64MatrixToFloat32")
	}
	
	for i := range flatArray {
		if flatArray[i] != expected[i] {
			t.Errorf("Value mismatch at index %d", i)
		}
	}
	
	// Test reverse conversion
	reconstructed := Float32ArrayToFloat64Matrix(flatArray, 2, 3)
	
	if len(reconstructed) != len(matrix) {
		t.Errorf("Row count mismatch in Float32ArrayToFloat64Matrix")
	}
	
	for i := range reconstructed {
		if len(reconstructed[i]) != len(matrix[i]) {
			t.Errorf("Column count mismatch in row %d", i)
		}
		for j := range reconstructed[i] {
			if reconstructed[i][j] != matrix[i][j] {
				t.Errorf("Value mismatch at [%d][%d]", i, j)
			}
		}
	}
}