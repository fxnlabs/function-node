package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

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
		if rootLogger != nil {
			rootLogger.Fatal("failed to run app", zap.Error(err))
		} else {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}
