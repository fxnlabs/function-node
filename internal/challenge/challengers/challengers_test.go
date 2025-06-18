package challengers

import (
	"bytes"
	"crypto/ecdsa"
	"io"
	"math/big"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"
	mocks "github.com/fxnlabs/function-node/mocks/challengers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestMatrixMultiplicationChallenger_Execute(t *testing.T) {
	log := zap.NewNop()
	challenger := &MatrixMultiplicationChallenger{}

	t.Run("valid multiplication", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		expected := map[string]interface{}{
			"C": [][]float64{{19, 22}, {43, 50}},
		}

		result, err := challenger.Execute(payload, log)
		assert.NoError(t, err)
		assert.Equal(t, expected, result)
	})

	t.Run("incompatible dimensions", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}},
			"B": [][]float64{{3, 4, 5}},
		}

		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("empty matrices", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{},
			"B": [][]float64{},
		}

		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid payload", func(t *testing.T) {
		payload := "invalid"
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid matrix A", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": "invalid",
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid matrix B", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
			"B": "invalid",
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("missing matrix A", func(t *testing.T) {
		payload := map[string]interface{}{
			"B": [][]float64{{5, 6}, {7, 8}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("missing matrix B", func(t *testing.T) {
		payload := map[string]interface{}{
			"A": [][]float64{{1, 2}, {3, 4}},
		}
		result, err := challenger.Execute(payload, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestNewEndpointReachableChallenger(t *testing.T) {
	challenger := NewEndpointReachableChallenger()
	assert.NotNil(t, challenger)
	assert.NotNil(t, challenger.Client)
}

func TestNewIdentityChallenger(t *testing.T) {
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)
	challenger := NewIdentityChallenger(privateKey)
	assert.NotNil(t, challenger)
	assert.Equal(t, privateKey, challenger.privateKey)
	assert.NotNil(t, challenger.Client)
	assert.NotNil(t, challenger.exec)
}

func TestCommand(t *testing.T) {
	executor := &CMDExecutor{}
	cmd := executor.Command("echo", "hello")
	assert.NotNil(t, cmd)
	assert.Equal(t, "echo", cmd.Args[0])
	assert.Equal(t, "hello", cmd.Args[1])
}

type mockRoundTripper struct {
	roundTripFunc func(req *http.Request) (*http.Response, error)
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.roundTripFunc(req)
}

func newTestClient(fn func(req *http.Request) (*http.Response, error)) *http.Client {
	return &http.Client{
		Transport: &mockRoundTripper{
			roundTripFunc: fn,
		},
	}
}

func TestEndpointReachableChallenger_Execute(t *testing.T) {
	log := zap.NewNop()

	t.Run("reachable endpoint", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       http.NoBody,
			}, nil
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.NoError(t, err)
		assert.Equal(t, map[string]bool{"reachable": true}, result)
	})

	t.Run("unreachable endpoint", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       http.NoBody,
			}, nil
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.NoError(t, err)
		assert.Equal(t, map[string]bool{"reachable": false}, result)
	})

	t.Run("http get error", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return nil, assert.AnError
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid payload", func(t *testing.T) {
		challenger := &EndpointReachableChallenger{Client: &http.Client{}}
		result, err := challenger.Execute(123, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestIdentityChallenger_Execute(t *testing.T) {
	log := zap.NewNop()
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)

	mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("127.0.0.1")),
		}, nil
	})

	t.Run("default", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "system_profiler", "SPDisplaysDataType").Return(exec.Command("echo", "Graphics/Displays:"))
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "Pages free: 200."))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "darwin"
		result, err := challenger.Execute(nil, log)
		assert.NoError(t, err)
		assert.NotNil(t, result)

		resMap, ok := result.(map[string]interface{})
		assert.True(t, ok)
		assert.NotEmpty(t, resMap["publicKey"])
		assert.Equal(t, "127.0.0.1", resMap["ipAddress"])
		assert.NotNil(t, resMap["signature"])
	})

	t.Run("ip address fetch error", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return nil, assert.AnError
		})
		mockExec := new(mocks.MockExecutor)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		result, err := challenger.Execute(nil, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("ip address read error", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(&errorReader{}),
			}, nil
		})
		mockExec := new(mocks.MockExecutor)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		result, err := challenger.Execute(nil, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("crypto sign error", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				// Expected panic
			}
		}()
		badPrivateKey := &ecdsa.PrivateKey{
			PublicKey: ecdsa.PublicKey{
				Curve: nil,
				X:     nil,
				Y:     nil,
			},
			D: new(big.Int),
		}
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "system_profiler", "SPDisplaysDataType").Return(exec.Command("echo", "Graphics/Displays:"))
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "Pages free: 200."))
		challenger := &IdentityChallenger{
			privateKey: badPrivateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "darwin"
		result, err := challenger.Execute(nil, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("get gpu stats error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "system_profiler", "SPDisplaysDataType").Return(exec.Command("false"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "darwin"
		result, err := challenger.Execute(nil, log)
		assert.NoError(t, err)
		resMap, ok := result.(map[string]interface{})
		assert.True(t, ok)
		assert.Nil(t, resMap["gpuStats"])
	})

	t.Run("get gpu stats error linux", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("false"))
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(exec.Command("false"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "linux"
		result, err := challenger.Execute(nil, log)
		assert.NoError(t, err)
		resMap, ok := result.(map[string]interface{})
		assert.True(t, ok)
		assert.Nil(t, resMap["gpuStats"])
	})

	t.Run("parse gpu stats unsupported", func(t *testing.T) {
		_, err := parseGPUStats(nil, "unsupported", log, nil)
		assert.Error(t, err)
	})

	t.Run("getNvidiaGPUStats not found", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		cmd := exec.Command("sh", "-c", "echo 'executable file not found' >&2; exit 1")
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(cmd)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getNvidiaGPUStats(log)
		assert.Error(t, err)
		assert.EqualError(t, err, "nvidia-smi not found")
	})

	t.Run("getNvidiaGPUStats error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		cmd := exec.Command("false")
		cmd.Stderr = &bytes.Buffer{}
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(cmd)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getNvidiaGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getNvidiaGPUStats generic error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("non-existent-command"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getNvidiaGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getNvidiaGPUStats success", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		expectedOutput := "GeForce RTX 3080, 470.57.02, 10240, 1024, 9216, 90, 50"
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("echo", expectedOutput))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		output, err := challenger.getNvidiaGPUStats(log)
		assert.NoError(t, err)
		assert.Equal(t, expectedOutput+"\n", string(output))
	})

	t.Run("getNvidiaGPUStats generic error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("non-existent-command"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getNvidiaGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getAmdGPUStats not found", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		cmd := exec.Command("sh", "-c", "echo 'executable file not found' >&2; exit 1")
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(cmd)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getAmdGPUStats(log)
		assert.Error(t, err)
		assert.EqualError(t, err, "rocm-smi not found")
	})

	t.Run("getAmdGPUStats error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		cmd := exec.Command("false")
		cmd.Stderr = &bytes.Buffer{}
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(cmd)
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getAmdGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getAmdGPUStats generic error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(exec.Command("non-existent-command"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getAmdGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getAmdGPUStats success", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		expectedOutput := "card0, Radeon RX 6800, 16384, 1024"
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(exec.Command("echo", expectedOutput))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		output, err := challenger.getAmdGPUStats(log)
		assert.NoError(t, err)
		assert.Equal(t, expectedOutput+"\n", string(output))
	})

	t.Run("getAmdGPUStats generic error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(exec.Command("non-existent-command"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getAmdGPUStats(log)
		assert.Error(t, err)
	})

	t.Run("getMacSystemMemory error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("false"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getMacSystemMemory(log)
		assert.Error(t, err)
	})

	t.Run("getMacSystemMemory parse error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "invalid"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, err := challenger.getMacSystemMemory(log)
		assert.Error(t, err)
	})

	t.Run("getMacMemoryUsage error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("false"))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.Error(t, err)
	})

	t.Run("parseNvidiaGPUStats", func(t *testing.T) {
		output, err := os.ReadFile("../../../fixtures/tests/nvidia_smi.txt")
		require.NoError(t, err)
		stats, err := parseNvidiaGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "GeForce RTX 3080", stats[0].Name)
	})

	t.Run("parseNvidiaGPUStats invalid", func(t *testing.T) {
		output := []byte("invalid")
		stats, err := parseNvidiaGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 0)
	})

	t.Run("parseAmdGPUStats", func(t *testing.T) {
		output, err := os.ReadFile("../../../fixtures/tests/rocm_smi.txt")
		require.NoError(t, err)
		stats, err := parseAmdGPUStats(output, log)
		assert.NoError(t, err)
		if assert.Len(t, stats, 1) {
			assert.Equal(t, "Radeon RX 6800", stats[0].Name)
		}
	})

	t.Run("parseAmdGPUStats invalid", func(t *testing.T) {
		output := []byte("invalid")
		stats, err := parseAmdGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 0)
	})

	t.Run("parseAmdGPUStats invalid number", func(t *testing.T) {
		output := []byte("card, name, vram_total, vram_used\ncard0,Radeon RX 6800,invalid,1024")
		stats, err := parseAmdGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, 0, stats[0].VRAMTotalMB)
	})

	t.Run("parseAmdGPUStats invalid line", func(t *testing.T) {
		output := []byte("card, name, vram_total, vram_used\ncard0,Radeon RX 6800")
		stats, err := parseAmdGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 0)
	})

	t.Run("parseMacGPUStats", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): 8 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		vmStatOutput, err := os.ReadFile("../../../fixtures/tests/vm_stat.txt")
		require.NoError(t, err)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", string(vmStatOutput)))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "Apple M1", stats[0].Name)
	})

	t.Run("parseMacGPUStats memory error", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): invalid")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: 200."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, err := challenger.parseMacGPUStats(output, log)
		assert.Error(t, err)
	})

	t.Run("parseMacGPUStats no driver version", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): 8 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("false"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: 200."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "N/A", stats[0].DriverVersion)
	})

	t.Run("parseMacGPUStats no vram", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: 200."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "Apple M1", stats[0].Name)
	})

	t.Run("getMacMemoryUsage page size error", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "invalid page size"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.Error(t, err)
	})

	t.Run("getMacMemoryUsage no free pages", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.NoError(t, err)
	})

	t.Run("getMacMemoryUsage no page size", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "invalid output"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.Error(t, err)
	})

	t.Run("getMacMemoryUsage invalid page value", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: invalid."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		used, free, err := challenger.getMacMemoryUsage(log)
		assert.NoError(t, err)
		assert.Equal(t, 0, used)
		assert.Equal(t, 0, free)
	})

	t.Run("getMacMemoryUsage invalid line", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\ninvalid"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.NoError(t, err)
	})

	t.Run("getMacMemoryUsage no page size line", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "invalid"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		_, _, err := challenger.getMacMemoryUsage(log)
		assert.Error(t, err)
	})

	t.Run("getGPUStats linux fallback", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("false"))
		rocmSmiOutput, err := os.ReadFile("../../../fixtures/tests/rocm_smi.txt")
		require.NoError(t, err)
		mockExec.On("Command", "rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "all", "--csv").Return(exec.Command("echo", string(rocmSmiOutput)))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "linux"
		_, err = challenger.getGPUStats(log)
		assert.NoError(t, err)
	})

	t.Run("getGPUStats unsupported os", func(t *testing.T) {
		mockExec := new(mocks.MockExecutor)
		nvidiaSmiOutput, err := os.ReadFile("../../../fixtures/tests/nvidia_smi.txt")
		require.NoError(t, err)
		mockExec.On("Command", "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits").Return(exec.Command("echo", string(nvidiaSmiOutput)))
		challenger := &IdentityChallenger{
			privateKey: privateKey,
			Client:     mockClient,
			exec:       mockExec,
		}
		challenger.goos = "unsupported"
		_, err = challenger.getGPUStats(log)
		assert.NoError(t, err)
	})

	t.Run("parseMacGPUStats with apple silicon", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): 8 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: 200."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "Apple M1", stats[0].Name)
		assert.True(t, stats[0].UnifiedMemory)
	})

	t.Run("parseMacGPUStats with multiple GPUs", func(t *testing.T) {
		output := []byte("Chipset Model: Intel Iris Pro\nVRAM (Total): 1536 MB\nChipset Model: NVIDIA GeForce GT 750M\nVRAM (Total): 2 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 2)
		assert.Equal(t, "Intel Iris Pro", stats[0].Name)
		assert.Equal(t, "NVIDIA GeForce GT 750M", stats[1].Name)
	})

	t.Run("parseMacGPUStats with getMacSystemMemory error", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): 8 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("false"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("echo", "page size of 4096 bytes\nPages free: 200."))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "Apple M1", stats[0].Name)
	})

	t.Run("parseMacGPUStats with getMacMemoryUsage error", func(t *testing.T) {
		output := []byte("Chipset Model: Apple M1\nVRAM (Total): 8 GB")
		mockExec := new(mocks.MockExecutor)
		mockExec.On("Command", "sw_vers", "-productVersion").Return(exec.Command("echo", "14.0"))
		mockExec.On("Command", "sysctl", "-n", "hw.memsize").Return(exec.Command("echo", "16384"))
		mockExec.On("Command", "vm_stat").Return(exec.Command("false"))
		challenger := &IdentityChallenger{
			exec: mockExec,
		}
		stats, err := challenger.parseMacGPUStats(output, log)
		assert.NoError(t, err)
		assert.Len(t, stats, 1)
		assert.Equal(t, "Apple M1", stats[0].Name)
	})

	t.Run("parseMemoryString", func(t *testing.T) {
		val, err := parseMemoryString("1024 MB")
		assert.NoError(t, err)
		assert.Equal(t, 1024, val)

		val, err = parseMemoryString("2 GB")
		assert.NoError(t, err)
		assert.Equal(t, 2048, val)

		val, err = parseMemoryString("1 TB")
		assert.NoError(t, err)
		assert.Equal(t, 1048576, val)

		val, err = parseMemoryString("1024")
		assert.NoError(t, err)
		assert.Equal(t, 1024, val)

		_, err = parseMemoryString("invalid")
		assert.Error(t, err)

		_, err = parseMemoryString("invalid GB")
		assert.Error(t, err)
	})
}

type errorReader struct{}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, assert.AnError
}
