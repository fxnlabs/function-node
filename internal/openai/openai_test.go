package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/fxnlabs/function-node/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestNewModelsHandler(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackendConfig{
		Models: map[string]config.ModelBackend{
			"model-1": {},
			"model-2": {},
		},
	}

	handler := NewModelsHandler(backendConfig, log)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)

	var modelList ModelList
	err := json.NewDecoder(rr.Body).Decode(&modelList)
	require.NoError(t, err)

	assert.Equal(t, "list", modelList.Object)
	assert.Len(t, modelList.Data, 2)

	modelIDs := make([]string, 0, len(modelList.Data))
	for _, m := range modelList.Data {
		modelIDs = append(modelIDs, m.ID)
	}
	assert.Contains(t, modelIDs, "model-1")
	assert.Contains(t, modelIDs, "model-2")
}

func TestNewModelsHandler_NoModels(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackendConfig{
		Models: map[string]config.ModelBackend{},
	}

	handler := NewModelsHandler(backendConfig, log)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)

	var modelList ModelList
	err := json.NewDecoder(rr.Body).Decode(&modelList)
	require.NoError(t, err)

	assert.Equal(t, "list", modelList.Object)
	assert.Len(t, modelList.Data, 0)
}

func TestNewOAIProxyHandler(t *testing.T) {
	log := zap.NewNop()

	// Create a mock backend server
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if the auth header was passed correctly
		assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))

		// Check that the body was proxied
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		assert.Equal(t, `{"model": "model-1"}`, string(body))

		w.WriteHeader(http.StatusOK)
		w.Write([]byte("proxied response"))
	}))
	defer backendServer.Close()

	backendConfig := &config.ModelBackendConfig{
		Models: map[string]config.ModelBackend{
			"model-1": {
				URL: backendServer.URL,
				Auth: &config.AuthConfig{
					BearerToken: "test-token",
				},
			},
		},
	}

	handler := NewOAIProxyHandler(backendConfig, log)

	t.Run("successful proxy", func(t *testing.T) {
		reqBody := `{"model": "model-1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusOK, rr.Code)
		respBody, err := io.ReadAll(rr.Body)
		require.NoError(t, err)
		assert.Equal(t, "proxied response", string(respBody))
	})

	t.Run("model not found", func(t *testing.T) {
		reqBody := `{"model": "unknown-model"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusBadRequest, rr.Code)
	})

	t.Run("invalid json body", func(t *testing.T) {
		reqBody := `{"model": "model-1"`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusBadRequest, rr.Code)
	})
}

func TestProxyRequest_APIKeyAuth(t *testing.T) {
	log := zap.NewNop()

	// Create a mock backend server
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "test-api-key", r.Header.Get("x-api-key"))
		w.WriteHeader(http.StatusOK)
	}))
	defer backendServer.Close()

	backendConfig := &config.ModelBackendConfig{
		Models: map[string]config.ModelBackend{
			"model-1": {
				URL: backendServer.URL,
				Auth: &config.AuthConfig{
					APIKey: "test-api-key",
				},
			},
		},
	}

	handler := NewOAIProxyHandler(backendConfig, log)
	reqBody := `{"model": "model-1"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
}

func TestProxyRequest_Errors(t *testing.T) {
	log := zap.NewNop()

	t.Run("invalid backend url", func(t *testing.T) {
		backendConfig := &config.ModelBackendConfig{
			Models: map[string]config.ModelBackend{
				"model-1": {
					URL: "invalid-url::",
				},
			},
		}
		handler := NewOAIProxyHandler(backendConfig, log)
		reqBody := `{"model": "model-1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		assert.Equal(t, http.StatusBadGateway, rr.Code)
	})

	t.Run("backend server error", func(t *testing.T) {
		backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer backendServer.Close()

		backendConfig := &config.ModelBackendConfig{
			Models: map[string]config.ModelBackend{
				"model-1": {
					URL: backendServer.URL,
				},
			},
		}
		handler := NewOAIProxyHandler(backendConfig, log)
		reqBody := `{"model": "model-1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		assert.Equal(t, http.StatusInternalServerError, rr.Code)
	})
}

func TestNewOAIProxyHandler_EmptyBody(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackendConfig{
		Models: map[string]config.ModelBackend{
			"default": {
				URL: "http://localhost",
			},
		},
	}
	handler := NewOAIProxyHandler(backendConfig, log)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(""))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadGateway, rr.Code)
}
