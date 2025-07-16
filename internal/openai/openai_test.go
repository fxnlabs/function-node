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
	backendConfig := &config.ModelBackend{
		FxnID: "fxn-id-1",
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
	assert.Len(t, modelList.Data, 1)
	assert.Equal(t, "fxn-id-1", modelList.Data[0].ID)
	assert.Equal(t, "fxn-id-1", modelList.Data[0].FxnID)
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
		assert.Equal(t, `{"model": "aliased-model-name"}`, string(body))

		w.WriteHeader(http.StatusOK)
		w.Write([]byte("proxied response"))
	}))
	defer backendServer.Close()

	backendConfig := &config.ModelBackend{
		URL:            backendServer.URL,
		BearerToken:    "test-token",
		ModelNameAlias: "aliased-model-name",
	}

	handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)

	t.Run("successful proxy", func(t *testing.T) {
		reqBody := `{"model": "model-1"}` // Note the space after the colon
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		assert.Equal(t, http.StatusOK, rr.Code)
		respBody, err := io.ReadAll(rr.Body)
		require.NoError(t, err)
		assert.Equal(t, "proxied response", string(respBody))
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

	backendConfig := &config.ModelBackend{
		URL:    backendServer.URL,
		APIKey: "test-api-key",
	}

	handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)
	reqBody := `{"model": "model-1"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
}

func TestProxyRequest_Errors(t *testing.T) {
	log := zap.NewNop()

	t.Run("invalid backend url", func(t *testing.T) {
		backendConfig := &config.ModelBackend{
			URL: "invalid-url::",
		}
		handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)
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

		backendConfig := &config.ModelBackend{
			URL: backendServer.URL,
		}
		handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)
		reqBody := `{"model": "model-1"}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		assert.Equal(t, http.StatusInternalServerError, rr.Code)
	})
}

func TestNewOAIProxyHandler_EmptyBody(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackend{
		URL: "http://localhost",
	}
	handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(""))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadGateway, rr.Code)
}

type errorReader struct{}

func (errorReader) Read(p []byte) (n int, err error) {
	return 0, assert.AnError
}

func TestNewOAIProxyHandler_BodyReadError(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackend{
		URL: "http://localhost",
	}
	handler := NewOAIProxyHandler(&config.Config{}, backendConfig, log)
	req := httptest.NewRequest("POST", "/v1/chat/completions", errorReader{})
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

type errorWriter struct{}

func (errorWriter) Header() http.Header {
	return http.Header{}
}

func (errorWriter) Write(p []byte) (n int, err error) {
	return 0, assert.AnError
}

func (errorWriter) WriteHeader(statusCode int) {}

func TestNewModelsHandler_EncoderError(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackend{}
	handler := NewModelsHandler(backendConfig, log)
	req := httptest.NewRequest("GET", "/v1/models", nil)
	rr := errorWriter{}
	handler.ServeHTTP(rr, req)
}

type errorTransport struct{}

func (t *errorTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, assert.AnError
}

func TestProxyRequest_ClientDoError(t *testing.T) {
	log := zap.NewNop()
	backendConfig := &config.ModelBackend{
		URL: "http://localhost",
	}

	// Create a custom client with the error-producing transport
	client := &http.Client{
		Transport: &errorTransport{},
	}

	reqBody := `{"model": "model-1"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	rr := httptest.NewRecorder()

	// We need to get the modelBackend to pass to proxyRequest
	modelBackend, err := backendConfig.GetModelBackend(req, log)
	require.NoError(t, err)

	proxyRequest(req, modelBackend, rr, client)

	assert.Equal(t, http.StatusBadGateway, rr.Code)
}
