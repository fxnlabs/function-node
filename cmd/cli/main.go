package main

import (
	"fmt"
	"os"

	"github.com/fxnlabs/function-node/internal/config"
	"github.com/fxnlabs/function-node/internal/keys"
	"github.com/fxnlabs/function-node/internal/logger"
	"github.com/urfave/cli/v2"
	"go.uber.org/zap"
)

func main() {
	var log *zap.Logger
	app := &cli.App{
		Name:  "fxn",
		Usage: "A CLI for interacting with the Function Network",
		Before: func(c *cli.Context) error {
			cfg, err := config.LoadConfig(c.String("config"))
			if err != nil {
				return err
			}
			zapLogger, err := logger.New(cfg.Logger.Verbosity)
			if err != nil {
				return err
			}
			log = zapLogger.Named("cli")
			return nil
		},
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "config",
				Value:   "config.yaml",
				Usage:   "Load configuration from `FILE`",
				EnvVars: []string{"FXN_CONFIG"},
			},
		},
		Commands: []*cli.Command{
			{
				Name:  "account",
				Usage: "Manage account",
				Subcommands: []*cli.Command{
					{
						Name:  "new",
						Usage: "Create a new account",
						Flags: []cli.Flag{
							&cli.StringFlag{
								Name:  "out",
								Value: "nodekey.json",
								Usage: "path to save the key file",
							},
						},
						Action: func(c *cli.Context) error {
							return keys.GenerateKeyFile(c.String("out"))
						},
					},
					{
						Name:  "get",
						Usage: "Get the account address",
						Flags: []cli.Flag{
							&cli.StringFlag{
								Name:  "in",
								Value: "nodekey.json",
								Usage: "path to the key file",
							},
						},
						Action: func(c *cli.Context) error {
							_, address, err := keys.LoadPrivateKey(c.String("in"))
							if err != nil {
								return err
							}
							log.Info("Account address", zap.String("address", address.Hex()))
							return nil
						},
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		if log != nil {
			log.Fatal("failed to run app", zap.Error(err))
		} else {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}
