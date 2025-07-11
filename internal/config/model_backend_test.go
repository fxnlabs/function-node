package config

import (
	"bytes"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestLoadModelBackendConfig(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		config, err := LoadModelBackend("../../fixtures/tests/config/valid_model_backend.yaml")
		require.NoError(t, err)
		require.NotNil(t, config)

		assert.Equal(t, "fxn", config.BackendProvider)
		assert.Equal(t, "default", config.FxnID)
		assert.Equal(t, "default-api-key", config.APIKey)
	})

	t.Run("non-existent file", func(t *testing.T) {
		config, err := LoadModelBackend("non-existent-file.yaml")
		assert.Error(t, err)
		assert.Nil(t, config)
	})

	t.Run("invalid yaml", func(t *testing.T) {
		config, err := LoadModelBackend("../../fixtures/tests/config/invalid_config.yaml")
		assert.Error(t, err)
		assert.Nil(t, config)
	})
}

func TestGetModelBackend(t *testing.T) {
	log := zap.NewNop()
	config, err := LoadModelBackend("../../fixtures/tests/config/valid_model_backend.yaml")
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/", bytes.NewBufferString(`{"model": "default"}`))
	require.NoError(t, err)

	modelBackend, err := config.GetModelBackend(req, log)
	require.NoError(t, err)
	assert.Equal(t, "fxn", modelBackend.BackendProvider)
	assert.Equal(t, "default", modelBackend.FxnID)
	assert.Equal(t, "default-api-key", modelBackend.APIKey)
}
