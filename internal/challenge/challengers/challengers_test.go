package challengers

import (
	"io"
	"net/http"
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

	result, err := challenger.Execute(nil, log)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	resMap, ok := result.(map[string]interface{})
	assert.True(t, ok)
	assert.NotEmpty(t, resMap["publicKey"])
	assert.Equal(t, "127.0.0.1", resMap["ipAddress"])
	assert.NotNil(t, resMap["signature"])
}
