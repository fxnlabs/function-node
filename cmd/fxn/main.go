package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/fxnlabs/function-node/fixtures"
	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func ensureDefaultConfigs(configHomePath string) error {
	// Create the config directory if it doesn't exist
	if err := os.MkdirAll(configHomePath, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	templateFiles := []struct {
		name    string
		content []byte
	}{
		{"model_backend.yaml", fixtures.ModelBackendTemplate},
		{"config.yaml", fixtures.ConfigTemplate},
	}

	for _, file := range templateFiles {
		path := filepath.Join(configHomePath, file.name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := os.WriteFile(path, file.content, 0644); err != nil {
				return fmt.Errorf("failed to write default %s: %w", file.name, err)
			}
		}
	}

	nodeKeyPath := filepath.Join(configHomePath, "nodekey.json")
	if _, err := os.Stat(nodeKeyPath); os.IsNotExist(err) {
		if err := keys.GenerateKeyFile(nodeKeyPath); err != nil {
			return fmt.Errorf("failed to generate node key: %w", err)
		}
	}
	return nil
}

func main() {
	var home string
	var cfg *config.Config
	var zapLogger *zap.Logger
	var rootLogger *zap.Logger

	app := &cli.App{
		Name:     "fxn",
		Metadata: map[string]interface{}{},
		Usage:    "A CLI for interacting with the Function Network",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:        "home",
				Value:       config.GetDefaultConfigHome(),
				Usage:       "Path to the fxn home directory",
				EnvVars:     []string{"home"},
				Destination: &home,
			},
		},
		Before: func(c *cli.Context) error {
			if err := ensureDefaultConfigs(home); err != nil {
				return fmt.Errorf("failed to set up default configuration: %w", err)
			}

			var err error
			cfg, err = config.LoadConfig(filepath.Join(home, "config.yaml"))
			if err != nil {
				return err
			}
			zapLogger, err = logger.New(cfg.Logger.Verbosity)
			if err != nil {
				return err
			}
			rootLogger := zapLogger.Named("cli")
			// store them for downstream commands:
			c.App.Metadata["cfg"] = cfg
			c.App.Metadata["logger"] = rootLogger
			c.App.Metadata["homeDir"] = home
			return nil
		},
		Commands: []*cli.Command{
			accountCommands(),
			startCommand(),
		},
	}

	if err := app.Run(os.Args); err != nil {
		rootLogger.Fatal("failed to run app", zap.Error(err))
	}
}
