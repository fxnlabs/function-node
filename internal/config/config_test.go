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
		config, err := LoadConfig("../../fixtures/tests/config/valid_config.yaml")
		require.NoError(t, err)
		require.NotNil(t, config)

		assert.Equal(t, 8080, config.Node.ListenPort)
		assert.Equal(t, "127.0.0.1", config.Node.ListenAddress)
		assert.Equal(t, "info", config.Logger.Verbosity)
		assert.Equal(t, "0x123", config.Registry.RouterSmartContractAddress)
		assert.Equal(t, 10*time.Second, config.Registry.Gateway.PollInterval)
		assert.Equal(t, 15*time.Second, config.Registry.Provider.PollInterval)
		assert.Equal(t, "http://localhost:8545", config.RpcProvider)
		assert.Equal(t, "0x456", config.SchedulerAddress)
		assert.Equal(t, 1*time.Minute, config.NonceCache.TTL)
		assert.Equal(t, 30*time.Second, config.NonceCache.CleanupInterval)
		assert.Equal(t, 50, config.Proxy.MaxIdleConns)
		assert.Equal(t, 60*time.Second, config.Proxy.IdleConnTimeout)
	})

	t.Run("non-existent file", func(t *testing.T) {
		_, err := LoadConfig("non-existent-file.yaml")
		assert.Error(t, err)
	})

	t.Run("invalid yaml", func(t *testing.T) {
		dir, err := os.Getwd()
		require.NoError(t, err)

		configPath := filepath.Join(dir, "..", "..", "fixtures", "tests", "invalid_config", "config.yaml")
		_, err = LoadConfig(configPath)
		assert.Error(t, err)
	})
}
