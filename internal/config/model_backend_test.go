package config

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestLoadModelBackendConfig(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		config, err := LoadModelBackendConfig("../../fixtures/tests/config/model_backend.yaml")
		require.NoError(t, err)
		require.NotNil(t, config)

		assert.Len(t, config.Models, 2)

		defaultModel, ok := config.Models["default"]
		require.True(t, ok)
		assert.Equal(t, "http://default-model-backend.com", defaultModel.URL)
		require.NotNil(t, defaultModel.Auth)
		assert.Equal(t, "default-api-key", defaultModel.Auth.APIKey)

		llamaModel, ok := config.Models["meta/llama-4-scout-17b-16e-instruct"]
		require.True(t, ok)
		assert.Equal(t, "http://llama-backend.com", llamaModel.URL)
		require.NotNil(t, llamaModel.Auth)
		assert.Equal(t, "llama-bearer-token", llamaModel.Auth.BearerToken)
	})

	t.Run("non-existent file", func(t *testing.T) {
		config, err := LoadModelBackendConfig("non-existent-file.yaml")
		assert.Error(t, err)
		assert.Nil(t, config)
	})

	t.Run("invalid yaml", func(t *testing.T) {
		content := "invalid-yaml"
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		err := os.WriteFile(configPath, []byte(content), 0644)
		require.NoError(t, err)

		config, err := LoadModelBackendConfig(configPath)
		assert.Error(t, err)
		assert.Nil(t, config)
	})
}

func TestGetModelBackend(t *testing.T) {
	log := zap.NewNop()
	config, err := LoadModelBackendConfig("../../fixtures/tests/config/model_backend.yaml")
	require.NoError(t, err)

	t.Run("valid model in request", func(t *testing.T) {
		body := `{"model": "meta/llama-4-scout-17b-16e-instruct"}`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		modelBackend, err := config.GetModelBackend(req, log)
		require.NoError(t, err)
		assert.Equal(t, "http://llama-backend.com", modelBackend.URL)
	})

	t.Run("model not in config, fallback to default", func(t *testing.T) {
		body := `{"model": "unknown-model"}`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		modelBackend, err := config.GetModelBackend(req, log)
		require.NoError(t, err)
		assert.Equal(t, "http://default-model-backend.com", modelBackend.URL)
	})

	t.Run("model not in config, no default", func(t *testing.T) {
		configWithoutDefault := &ModelBackendConfig{
			Models: map[string]ModelBackend{
				"meta/llama-4-scout-17b-16e-instruct": {
					URL: "http://llama-backend.com",
				},
			},
		}
		body := `{"model": "unknown-model"}`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		_, err = configWithoutDefault.GetModelBackend(req, log)
		assert.Error(t, err)
	})

	t.Run("default model not in config", func(t *testing.T) {
		configWithoutDefault := &ModelBackendConfig{
			Models: map[string]ModelBackend{
				"meta/llama-4-scout-17b-16e-instruct": {
					URL: "http://llama-backend.com",
				},
			},
		}
		body := `{"model": "default"}`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		_, err = configWithoutDefault.GetModelBackend(req, log)
		assert.Error(t, err)
	})

	t.Run("no request body", func(t *testing.T) {
		req, err := http.NewRequest("POST", "/", nil)
		require.NoError(t, err)

		_, err = config.GetModelBackend(req, log)
		assert.Error(t, err)
	})

	t.Run("error reading request body", func(t *testing.T) {
		errorReader := &errorReader{}
		req, err := http.NewRequest("POST", "/", errorReader)
		require.NoError(t, err)

		_, err = config.GetModelBackend(req, log)
		assert.Error(t, err)
	})

	t.Run("invalid json body", func(t *testing.T) {
		body := `invalid-json`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		modelBackend, err := config.GetModelBackend(req, log)
		require.NoError(t, err)
		assert.Equal(t, "http://default-model-backend.com", modelBackend.URL)
	})

	t.Run("request body is read twice", func(t *testing.T) {
		body := `{"model": "meta/llama-4-scout-17b-16e-instruct"}`
		req, err := http.NewRequest("POST", "/", bytes.NewBufferString(body))
		require.NoError(t, err)

		modelBackend, err := config.GetModelBackend(req, log)
		require.NoError(t, err)
		assert.Equal(t, "http://llama-backend.com", modelBackend.URL)

		// Try to read the body again to ensure it was replaced
		bodyBytes, err := io.ReadAll(req.Body)
		require.NoError(t, err)
		assert.Equal(t, body, string(bodyBytes))
	})
}

type errorReader struct{}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, assert.AnError
}
