package logger

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestNew(t *testing.T) {
	t.Run("valid verbosity level", func(t *testing.T) {
		logger, err := New("info")
		require.NoError(t, err)
		assert.NotNil(t, logger)
		assert.True(t, logger.Core().Enabled(zap.InfoLevel))
		assert.False(t, logger.Core().Enabled(zap.DebugLevel))
	})

	t.Run("another valid verbosity level", func(t *testing.T) {
		logger, err := New("debug")
		require.NoError(t, err)
		assert.NotNil(t, logger)
		assert.True(t, logger.Core().Enabled(zap.DebugLevel))
	})

	t.Run("invalid verbosity level", func(t *testing.T) {
		logger, err := New("invalid")
		require.Error(t, err)
		assert.Nil(t, logger)
	})

	t.Run("empty verbosity level", func(t *testing.T) {
		// zap defaults to info level on empty string
		logger, err := New("")
		require.NoError(t, err)
		assert.NotNil(t, logger)
		assert.True(t, logger.Core().Enabled(zap.InfoLevel))
		assert.False(t, logger.Core().Enabled(zap.DebugLevel))
	})
}
