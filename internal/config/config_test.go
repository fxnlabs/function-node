package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadConfig(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		config, err := LoadConfig("../../fixtures/tests/config/config.yaml")
		require.NoError(t, err)
		require.NotNil(t, config)

		assert.Equal(t, "key.json", config.Node.Keyfile)
		assert.Equal(t, 8080, config.Node.ListenPort)
		assert.Equal(t, "info", config.Logger.Verbosity)
		assert.Equal(t, "0x123", config.Registry.RouterSmartContractAddress)
		assert.Equal(t, 10*time.Second, config.Registry.Gateway.PollInterval)
		assert.Equal(t, 15*time.Second, config.Registry.Provider.PollInterval)
		assert.Equal(t, "http://localhost:8545", config.RpcProvider)
		assert.Equal(t, "0x456", config.SchedulerAddress)
		assert.Equal(t, "model_backend.yaml", config.ModelBackendPath)
		assert.Equal(t, 1*time.Minute, config.NonceCache.TTL)
		assert.Equal(t, 30*time.Second, config.NonceCache.CleanupInterval)
	})

	t.Run("non-existent file", func(t *testing.T) {
		config, err := LoadConfig("non-existent-file.yaml")
		assert.Error(t, err)
		assert.Nil(t, config)
	})

	t.Run("invalid yaml", func(t *testing.T) {
		content := "invalid-yaml"
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		err := os.WriteFile(configPath, []byte(content), 0644)
		require.NoError(t, err)

		config, err := LoadConfig(configPath)
		assert.Error(t, err)
		assert.Nil(t, config)
	})
}
