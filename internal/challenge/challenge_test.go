package challenge

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/fxnlabs/function-node/internal/challenge/challengers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestNewChallenger(t *testing.T) {
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)

	testCases := []struct {
		name          string
		challengeType string
		expectedType  interface{}
		expectError   bool
	}{
		{
			name:          "matrix multiplication",
			challengeType: "MATRIX_MULTIPLICATION",
			expectedType:  &challengers.MatrixMultiplicationChallenger{},
			expectError:   false,
		},
		{
			name:          "endpoint reachable",
			challengeType: "ENDPOINT_REACHABLE",
			expectedType:  &challengers.EndpointReachableChallenger{},
			expectError:   false,
		},
		{
			name:          "identity",
			challengeType: "IDENTITY",
			expectedType:  challengers.NewIdentityChallenger(privateKey),
			expectError:   false,
		},
		{
			name:          "unknown",
			challengeType: "UNKNOWN",
			expectedType:  nil,
			expectError:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			challenger, err := NewChallenger(tc.challengeType, privateKey)
			if tc.expectError {
				assert.Error(t, err)
				assert.Nil(t, challenger)
			} else {
				assert.NoError(t, err)
				assert.IsType(t, tc.expectedType, challenger)
			}
		})
	}
}

type mockChallenger struct {
	executeFunc func(payload interface{}, log *zap.Logger) (interface{}, error)
}

func (m *mockChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	return m.executeFunc(payload, log)
}

func TestChallengeHandler(t *testing.T) {
	log := zap.NewNop()
	privateKey, err := crypto.GenerateKey()
	require.NoError(t, err)

	handler := ChallengeHandler(log, privateKey)

	t.Run("valid challenge", func(t *testing.T) {
		challenge := Challenge{
			Type:    "IDENTITY",
			Payload: nil,
		}
		body, _ := json.Marshal(challenge)
		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		rr := httptest.NewRecorder()

		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusOK, rr.Code)
	})

	t.Run("invalid request body", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/", bytes.NewReader([]byte("invalid json")))
		rr := httptest.NewRecorder()

		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusBadRequest, rr.Code)
	})

	t.Run("unknown challenge type", func(t *testing.T) {
		challenge := Challenge{
			Type:    "UNKNOWN",
			Payload: nil,
		}
		body, _ := json.Marshal(challenge)
		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		rr := httptest.NewRecorder()

		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusBadRequest, rr.Code)
	})

	t.Run("challenger execution error", func(t *testing.T) {
		challenge := Challenge{
			Type:    "ENDPOINT_REACHABLE",
			Payload: 123, // Invalid payload to cause execution error
		}
		body, _ := json.Marshal(challenge)
		req := httptest.NewRequest("POST", "/", bytes.NewReader(body))
		rr := httptest.NewRecorder()

		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusInternalServerError, rr.Code)
	})
}
