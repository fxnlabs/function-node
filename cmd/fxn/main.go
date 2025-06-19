package main

import (
	"fmt"
	"os"

	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func main() {
	cfg, err := config.LoadConfig("config.yaml")
	if err != nil {
		panic(err)
	}
	zapLogger, err := logger.New(cfg.Logger.Verbosity)
	if err != nil {
		panic(err)
	}
	rootLogger := zapLogger.Named("cli")
	app := &cli.App{
		Name:  "fxn",
		Usage: "A CLI for interacting with the Function Network",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "config",
				Value:   "config.yaml",
				Usage:   "Load configuration from `FILE`",
				EnvVars: []string{"FXN_CONFIG"},
			},
		},
		Commands: []*cli.Command{
			accountCommands(rootLogger),
			startCommand(rootLogger, cfg),
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
