package logger

import (
	"go.uber.org/zap"
)

func New(verbosity string) (*zap.Logger, error) {
	config := zap.NewProductionConfig()
	level, err := zap.ParseAtomicLevel(verbosity)
	if err != nil {
		return nil, err
	}
	config.Level = level
	return config.Build()
}
